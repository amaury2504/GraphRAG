// Hook PostToolUse (Bash, PowerShell) : après un `git push` ou un `gh pr create` réussi, ouvre une
// fenêtre de terminal qui affiche l'avancement de la CI en direct (scripts/ci_watch.py --live) :
// pourcentage global et par job. La fenêtre reste ouverte à la fin (-NoExit). Aucune intervention
// de Claude : le hook lance lui-même la fenêtre, détachée de la session.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";

let input;
try {
  input = JSON.parse(readFileSync(0, "utf8"));
} catch {
  process.exit(0);
}

const command = String(input?.tool_input?.command ?? "");
const declencheur = /\bgit\s+push\b/.test(command) || /\bgh\s+pr\s+create\b/.test(command);
const echec = input?.tool_response?.is_error === true || input?.tool_response?.interrupted === true;
if (!declencheur || echec || /ci_watch\.py/.test(command)) process.exit(0);

if (process.platform === "win32") {
  const suivi = spawn(
    "cmd.exe",
    ["/c", "start", "Suivi CI", "powershell", "-NoExit", "-Command", "uv run python scripts/ci_watch.py --live"],
    { cwd: input.cwd || process.cwd(), detached: true, stdio: "ignore" },
  );
  suivi.unref();
}
process.exit(0);
