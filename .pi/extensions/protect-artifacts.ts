/**
 * Blocks writes/edits to build output, caches, profiling artifacts, and the
 * committed dictionary binaries (which must be regenerated via the CLI).
 *
 * Also warns when the engine (src/lib.rs) or its Python bindings (crates/py)
 * are edited, since the Python extension must be rebuilt with
 * `uv run maturin develop --release` before the change takes effect.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BLOCKED = [
	{ test: /(^|\/)target\//, why: "Cargo build output" },
	{ test: /(^|\/)\.venv\//, why: "virtualenv (managed by uv sync)" },
	{ test: /(^|\/)__pycache__\//, why: "Python bytecode cache" },
	{ test: /(^|\/)\.(mypy|ruff)_cache\//, why: "linter cache" },
	{ test: /(^|\/)profile[^/]*\.(json\.gz|svg)$/, why: "generated profile artifact" },
	{ test: /\.pstats$/, why: "generated profile artifact" },
	{
		test: /(^|\/)words\/(dawg|gaddag)\.bin$/,
		why: "compiled dictionary; regenerate with `cargo run --release -- build` / `build-gaddag`",
	},
];

export default function (pi: ExtensionAPI) {
	let warnedAboutRebuild = false;

	pi.on("session_start", () => {
		warnedAboutRebuild = false;
	});

	pi.on("tool_call", async (event, ctx) => {
		if (event.toolName !== "write" && event.toolName !== "edit") return undefined;

		const path = String(event.input.path ?? "");
		const hit = BLOCKED.find((rule) => rule.test.test(path));

		if (hit) {
			if (ctx.hasUI) ctx.ui.notify(`Blocked write to ${path} (${hit.why})`, "warning");
			return { block: true, reason: `"${path}" is protected: ${hit.why}` };
		}

		if (!warnedAboutRebuild && /(^|\/)(src\/lib\.rs|crates\/py\/src\/.*\.rs)$/.test(path)) {
			warnedAboutRebuild = true;
			if (ctx.hasUI) {
				ctx.ui.notify(`${path} changed — run \`uv run maturin develop --release\``, "info");
			}
		}

		return undefined;
	});
}
