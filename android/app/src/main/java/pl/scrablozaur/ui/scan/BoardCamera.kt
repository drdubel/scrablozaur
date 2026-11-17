package pl.scrablozaur.ui.scan

import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import pl.scrablozaur.core.LOG_TAG
import pl.scrablozaur.engine.CameraFrame
import java.util.concurrent.Executor
import kotlin.math.max

/**
 * Frames the scanner reads are 1080p, as `frame-bench` times them: a board
 * filling most of the frame's width gets ~60 pixels a square, the cell
 * reader's 32 with room to spare.
 */
private val ANALYSIS_SIZE = Size(1920, 1080)

/**
 * The camera's preview with the grid the scanner found drawn on it. Every frame
 * goes to `onFrame` on `executor`, RGBA as the sensor gives it, with its time, the
 * lens's focal length in frame pixels, and a map from frame pixels to the
 * preview's.
 */
@Composable
fun BoardCamera(
    executor: Executor,
    quad: List<Offset>,
    color: Color,
    torch: Boolean,
    onFrame: (frame: CameraFrame, timeMs: Double, focalPx: Float, toView: (FloatArray) -> List<Offset>) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val controller = remember { LifecycleCameraController(context) }
    val handler by rememberUpdatedState(onFrame)
    val analyzer = remember { BoardAnalyzer(controller) { f, t, focal, toView -> handler(f, t, focal, toView) } }

    DisposableEffect(lifecycleOwner) {
        controller.imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
        controller.imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
        controller.imageAnalysisResolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy(ANALYSIS_SIZE, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build()
        controller.setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
        // A zoomed frame is a narrower lens than the one the scanner is told of.
        controller.isPinchToZoomEnabled = false
        controller.setImageAnalysisAnalyzer(executor, analyzer)
        controller.bindToLifecycle(lifecycleOwner)
        onDispose {
            controller.clearImageAnalysisAnalyzer()
            controller.unbind()
        }
    }
    LaunchedEffect(torch) { controller.enableTorch(torch) }

    Box(modifier) {
        AndroidView(
            factory = { PreviewView(it).apply { this.controller = controller; scaleType = PreviewView.ScaleType.FILL_CENTER } },
            modifier = Modifier.fillMaxSize(),
        )
        Canvas(Modifier.fillMaxSize()) {
            if (quad.size != 4) return@Canvas
            val path = Path().apply {
                moveTo(quad[0].x, quad[0].y)
                for (p in quad.drop(1)) lineTo(p.x, p.y)
                close()
            }
            drawPath(path, color, style = Stroke(width = 3.dp.toPx()))
        }
    }
}

/** Hands frames to the scanner, and knows where they land on the preview. */
private class BoardAnalyzer(
    private val controller: CameraController,
    private val onFrame: (CameraFrame, Double, Float, (FloatArray) -> List<Offset>) -> Unit,
) : ImageAnalysis.Analyzer {
    /** Sensor pixels to the preview's, from CameraX; null while the preview is not laid out. */
    @Volatile private var sensorToView: Matrix? = null
    private var buffer = ByteArray(0)
    /** The focal length for frames of a size, once the camera has said. */
    @Volatile private var focal: Pair<Size, Float>? = null
    @Volatile private var asked: Size? = null
    private val main = Handler(Looper.getMainLooper())

    override fun getTargetCoordinateSystem() = ImageAnalysis.COORDINATE_SYSTEM_VIEW_REFERENCED

    override fun updateTransform(matrix: Matrix?) {
        sensorToView = matrix?.let { Matrix(it) }
    }

    override fun analyze(image: ImageProxy) {
        image.use {
            val plane = image.planes[0]
            val stride = plane.rowStride
            val size = stride * image.height
            // One buffer, reused: a frame is 8 MB, and thirty of them a second
            // would keep the garbage collector busy.
            if (buffer.size != size) buffer = ByteArray(size)
            val pixels = plane.buffer
            pixels.rewind()
            pixels.get(buffer, 0, minOf(pixels.remaining(), size))
            val frame = CameraFrame(buffer, image.width.toUInt(), image.height.toUInt(), stride.toUInt())
            // Frame pixels to the preview's: back from the frame to the sensor, then on.
            val frameToView = sensorToView?.let { toView ->
                Matrix().takeIf { Matrix(image.imageInfo.sensorToBufferTransformMatrix).invert(it) }?.apply { postConcat(toView) }
            }
            val toView = { corners: FloatArray ->
                if (frameToView == null) {
                    emptyList()
                } else {
                    frameToView.mapPoints(corners)
                    corners.toList().chunked(2) { (x, y) -> Offset(x, y) }
                }
            }
            onFrame(frame, image.imageInfo.timestamp / 1e6, focalPx(Size(image.width, image.height)), toView)
        }
    }

    /** The camera's facts are read on the main thread, where CameraX wants them; until then, a guess. */
    private fun focalPx(size: Size): Float {
        focal?.let { (s, f) -> if (s == size) return f }
        if (asked != size) {
            asked = size
            main.post {
                val info = controller.cameraInfo
                if (info == null) {
                    asked = null
                    return@post
                }
                val f = focalPixels(info, size.width, size.height) ?: guess(size)
                Log.i(LOG_TAG, "scanner frames ${size.width}x${size.height}, focal length $f px")
                focal = size to f
            }
        }
        return guess(size)
    }

    private fun guess(size: Size) = 0.64f * max(size.width, size.height)
}

/**
 * The lens's focal length in pixels of a `width` x `height` frame, as the
 * sensor lies. The frame is the sensor's active array scaled to cover it and
 * cropped to the frame's shape. From the lens's calibration where the camera
 * publishes one, else from its focal length and the sensor's size; failing
 * both, a phone's usual main camera, 23 mm in 35 mm terms.
 */
@OptIn(ExperimentalCamera2Interop::class)
private fun focalPixels(info: CameraInfo, width: Int, height: Int): Float? {
    val camera = Camera2CameraInfo.from(info)
    val active = camera.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return null
    val scale = max(width / active.width().toFloat(), height / active.height().toFloat())
    camera.getCameraCharacteristic(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
        ?.takeIf { it.size >= 2 && it[0] > 0f }
        ?.let { return it[0] * scale }
    val mm = camera.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return null
    val sensor = camera.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
    val pixels = camera.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE) ?: return null
    return mm * pixels.width / sensor.width * scale
}
