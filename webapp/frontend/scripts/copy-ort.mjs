// Copia o runtime WASM do onnxruntime-web para public/ort/ antes do build e do dev server.
// visionOrt.ts carrega o runtime de "/ort/" (ort.env.wasm.wasmPaths). Os arquivos somam ~15 MB e
// ficam fora do Git (public/ort/ no .gitignore); copiar do node_modules a cada build garante que o
// runtime exista num checkout limpo (CI e Docker) e tenha sempre a mesma versao do pacote instalado.
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const origem = join(root, "node_modules", "onnxruntime-web", "dist");
const destino = join(root, "public", "ort");
const arquivos = ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

mkdirSync(destino, { recursive: true });
for (const nome of arquivos) {
  const de = join(origem, nome);
  if (!existsSync(de)) {
    console.error(`[copy-ort] ${de} nao existe: rode "npm ci" antes do build.`);
    process.exit(1);
  }
  copyFileSync(de, join(destino, nome));
  console.log(`[copy-ort] ${nome} (${(statSync(de).size / 1048576).toFixed(1)} MiB) -> public/ort/`);
}
