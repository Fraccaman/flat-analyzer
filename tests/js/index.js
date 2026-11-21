// Entry point so `node --test tests/js/` works (Node >=22 treats the arg as a
// file/module path, which resolves to this index). Imports every *.test.mjs.
const { readdirSync } = require("node:fs");
const { join } = require("node:path");
const { pathToFileURL } = require("node:url");
for (const f of readdirSync(__dirname).filter((n) => n.endsWith(".test.mjs")).sort()) {
  import(pathToFileURL(join(__dirname, f)).href);
}
