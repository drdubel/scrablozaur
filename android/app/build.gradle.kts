import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.io.File
import java.security.MessageDigest
import java.util.Properties
import javax.inject.Inject

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// The repository this Android project lives in: the engine, the language
// definitions and every trained artifact come from there, never from a copy.
val repoRoot: File = rootProject.projectDir.parentFile
val abiList = (findProperty("scrablozaur.abis") as String? ?: "arm64-v8a")
    .split(",").map { it.trim() }.filter { it.isNotEmpty() }

android {
    namespace = "pl.scrablozaur"
    // Compose 1.12 compiles against API 37; the app still targets 36.
    compileSdk = 37
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "pl.scrablozaur"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters.addAll(abiList) }
    }

    signingConfigs {
        // A sideloaded APK is updated by installing a newer one over it, which
        // Android only allows when both are signed with the same key. Keep
        // the keystore (and keystore.properties) out of git and backed up.
        val props = rootProject.file("keystore.properties")
        if (props.exists()) {
            val p = Properties().apply { props.inputStream().use { load(it) } }
            create("release") {
                storeFile = rootProject.file(p.getProperty("storeFile"))
                storePassword = p.getProperty("storePassword")
                keyAlias = p.getProperty("keyAlias")
                keyPassword = p.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    // Lexicons are memory-mapped or copied straight out of the APK, so they
    // are stored rather than deflated.
    androidResources { noCompress += "bin" }
    packaging { jniLibs { useLegacyPackaging = false } }
}

// ── The engine, built from the workspace ─────────────────────────────────────

/** Runs cargo with rustup's toolchain (Homebrew's rustc has no Android std). */
abstract class CargoTask : DefaultTask() {
    @get:Inject abstract val execOps: ExecOperations
    @get:Input abstract val toolchain: Property<String>
    @get:Input abstract val rustupBin: Property<String>
    @get:Internal abstract val workspace: DirectoryProperty

    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    val rustSources get() = workspace.asFileTree.matching {
        include("src/**/*.rs", "crates/**/*.rs", "**/Cargo.toml", "Cargo.lock", "crates/mobile/uniffi.toml")
        exclude("target/**", "android/**")
    }

    protected fun cargo(args: List<String>, extraEnv: Map<String, String> = emptyMap()) {
        execOps.exec {
            workingDir = workspace.get().asFile
            environment("PATH", "${rustupBin.get()}${File.pathSeparator}${System.getenv("PATH")}")
            environment("RUSTUP_TOOLCHAIN", toolchain.get())
            extraEnv.forEach { (k, v) -> environment(k, v) }
            commandLine(listOf("cargo") + args)
        }
    }
}

/** `libscrablozaur_mobile.so` for each ABI, cross-compiled with the NDK. */
abstract class BuildRustLibrary : CargoTask() {
    @get:Input abstract val abis: ListProperty<String>
    @get:Internal abstract val ndkDir: DirectoryProperty
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun build() {
        val triples = mapOf("arm64-v8a" to "aarch64-linux-android", "x86_64" to "x86_64-linux-android")
        val host = if (System.getProperty("os.name").startsWith("Mac")) "darwin-x86_64" else "linux-x86_64"
        val bin = ndkDir.get().asFile.resolve("toolchains/llvm/prebuilt/$host/bin")
        for (abi in abis.get()) {
            val triple = triples[abi] ?: error("no Rust target for ABI $abi")
            val key = triple.uppercase().replace('-', '_')
            val clang = bin.resolve("$triple${MIN_SDK}-clang").path
            cargo(
                listOf("build", "-p", "scrablozaur-mobile", "--profile", "mobile", "--target", triple),
                mapOf(
                    "CARGO_TARGET_${key}_LINKER" to clang,
                    "CC_${triple.replace('-', '_')}" to clang,
                    "AR_${triple.replace('-', '_')}" to bin.resolve("llvm-ar").path,
                    // 16 KB pages: required by newer devices, harmless on older.
                    "CARGO_TARGET_${key}_RUSTFLAGS" to "-C link-arg=-Wl,-z,max-page-size=16384",
                ),
            )
            val lib = workspace.get().asFile.resolve("target/$triple/mobile/libscrablozaur_mobile.so")
            lib.copyTo(outputDir.get().asFile.resolve("$abi/libscrablozaur_mobile.so"), overwrite = true)
        }
    }

    companion object {
        const val MIN_SDK = 26
    }
}

/** The Kotlin side of the UniFFI bindings, generated from a host build. */
abstract class GenerateBindings : CargoTask() {
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun generate() {
        cargo(listOf("build", "--release", "-p", "scrablozaur-mobile", "--features", "bindgen"))
        val ext = if (System.getProperty("os.name").startsWith("Mac")) "dylib" else "so"
        val root = workspace.get().asFile
        val out = outputDir.get().asFile
        out.deleteRecursively()
        execOps.exec {
            workingDir = root
            commandLine(
                root.resolve("target/release/uniffi-bindgen").path, "generate",
                "--library", root.resolve("target/release/libscrablozaur_mobile.$ext").path,
                "--language", "kotlin", "--no-format", "--out-dir", out.path,
            )
        }
    }
}

/**
 * The engine's data, staged from the repository with its repo-relative paths
 * intact -- so the paths inside `languages/<code>.json` resolve against the
 * extracted asset root exactly as they do against the checkout. A manifest
 * with sizes and SHA-256s lets the app verify what it extracted.
 */
abstract class StageEngineAssets : DefaultTask() {
    @get:Internal abstract val workspace: DirectoryProperty
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    val sources get() = workspace.asFileTree.matching {
        include(
            "languages/*.json", "words/*/dawg.bin", "words/*/gaddag.bin", "smart_player/models/*/leave_value.bin",
            "board_reader/src/models/*/grid_finder.bin", "board_reader/src/models/*/cell_reader.bin",
        )
    }

    @TaskAction
    fun stage() {
        val root = workspace.get().asFile
        val out = outputDir.get().asFile
        out.deleteRecursively()
        val files = mutableListOf<String>()
        for (def in root.resolve("languages").listFiles { f -> f.extension == "json" }!!.sorted()) {
            files += "languages/${def.name}"
            val json = JsonSlurper().parse(def) as Map<*, *>
            files += json["dawg"] as String
            files += json["gaddag"] as String
            (json["leave_net"] as Map<*, *>?)?.let { files += it["weights"] as String }
            (json["scanner"] as Map<*, *>?)?.let {
                files += it["grid_finder"] as String
                files += it["cell_reader"] as String
            }
        }
        val manifest = files.map { rel ->
            val src = root.resolve(rel)
            require(src.exists()) { "$rel is declared but missing" }
            src.copyTo(out.resolve("engine/$rel"), overwrite = true)
            val digest = MessageDigest.getInstance("SHA-256").digest(src.readBytes())
            mapOf("path" to rel, "size" to src.length(), "sha256" to digest.joinToString("") { "%02x".format(it) })
        }
        out.resolve("engine/manifest.json").writeText(JsonOutput.prettyPrint(JsonOutput.toJson(mapOf("files" to manifest))))
    }
}

val rustToolchainSetting = findProperty("scrablozaur.rustToolchain") as String? ?: "stable"
val rustupBinSetting = findProperty("scrablozaur.rustupBin") as String? ?: "${System.getProperty("user.home")}/.cargo/bin"

val buildRustLibrary = tasks.register<BuildRustLibrary>("buildRustLibrary") {
    workspace.set(repoRoot)
    toolchain.set(rustToolchainSetting)
    rustupBin.set(rustupBinSetting)
    abis.set(abiList)
    ndkDir.set(androidComponents.sdkComponents.ndkDirectory)
    outputDir.set(layout.buildDirectory.dir("generated/scrablozaur/jniLibs"))
}

val generateBindings = tasks.register<GenerateBindings>("generateBindings") {
    workspace.set(repoRoot)
    toolchain.set(rustToolchainSetting)
    rustupBin.set(rustupBinSetting)
    outputDir.set(layout.buildDirectory.dir("generated/scrablozaur/kotlin"))
}

val stageEngineAssets = tasks.register<StageEngineAssets>("stageEngineAssets") {
    workspace.set(repoRoot)
    outputDir.set(layout.buildDirectory.dir("generated/scrablozaur/assets"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.kotlin?.addGeneratedSourceDirectory(generateBindings, GenerateBindings::outputDir)
        variant.sources.jniLibs?.addGeneratedSourceDirectory(buildRustLibrary, BuildRustLibrary::outputDir)
        variant.sources.assets?.addGeneratedSourceDirectory(stageEngineAssets, StageEngineAssets::outputDir)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.compose.ui.tooling.preview)
    debugImplementation(libs.compose.ui.tooling)
    implementation(libs.kotlinx.coroutines.android)
    // UniFFI's Kotlin bindings call the engine through JNA.
    implementation("${libs.jna.get().module}:${libs.jna.get().version}@aar")
    testImplementation(libs.junit)
}
