/** The package version, written into every run row as `ctxdiff_version`. Kept
 * as a hand-maintained constant (not read from package.json at runtime) so the
 * bundled dist has zero filesystem/JSON-resolution dependency at import time.
 * Keep in sync with package.json's "version". */
export const VERSION = "0.1.0";
