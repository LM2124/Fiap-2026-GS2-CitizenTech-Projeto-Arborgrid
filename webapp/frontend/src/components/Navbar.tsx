import { ExternalLink } from "lucide-react";
import { APP_LABELS } from "../utils/labels";

export type AppTab = "bi" | "mapa" | "visao" | "assistente";

interface NavbarProps {
  activeTab: AppTab;
  onSelectTab: (tab: AppTab) => void;
  totalTrees?: number;
  totalViews?: number;
}

const NAV_ITEMS: { id: AppTab; label: string; shortLabel: string }[] = [
  { id: "bi", label: "Visão Geral", shortLabel: "Visão Geral" },
  { id: "mapa", label: "Mapa de Vistorias", shortLabel: "Mapa" },
  { id: "visao", label: "Análise de Imagem", shortLabel: "Imagem" },
  { id: "assistente", label: APP_LABELS.assistantTab, shortLabel: APP_LABELS.assistantTab },
];

export function Navbar({
  activeTab,
  onSelectTab,
  totalTrees = 0,
  totalViews = 0,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-40 border-b border-[#D5E0D1] bg-white">
      <div className="mx-auto flex max-w-[1680px] flex-wrap items-center justify-between gap-3 px-4 py-2.5 sm:px-8">
        <button
          type="button"
          onClick={() => onSelectTab("bi")}
          className="min-h-[44px] rounded py-1 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
        >
          <div className="flex items-baseline gap-2.5">
            <span className="text-lg font-bold tracking-tight text-[#1B4332]">
              ArborGrid SP
            </span>
            <span className="hidden text-xs font-medium text-[#4A5D4E] sm:inline">
              São Paulo
            </span>
          </div>
          <p className="text-xs text-[#4A5D4E]">
            Monitoramento urbano e prevenção de quedas na rede
          </p>
        </button>

        <nav
          aria-label="Navegação principal"
          className="flex flex-wrap items-center gap-1 rounded-md border border-[#DCE5D8] bg-[#F6F8F5] p-1"
        >
          {NAV_ITEMS.map((item) => {
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                aria-current={active ? "page" : undefined}
                onClick={() => onSelectTab(item.id)}
                className={`inline-flex min-h-[44px] items-center justify-center rounded px-3 py-2 text-[13px] font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F] sm:min-h-[38px] sm:py-1.5 sm:text-sm ${
                  active
                    ? "bg-[#1B4332] text-white"
                    : "text-[#2D4234] hover:bg-[#E7EFE4] hover:text-[#18241B]"
                }`}
              >
                <span className="hidden md:inline">{item.label}</span>
                <span className="md:hidden">{item.shortLabel}</span>
              </button>
            );
          })}
        </nav>

        <div className="hidden items-center gap-3 lg:flex">
          <span className="text-xs tabular-nums text-[#4A5D4E]">
            Amostra:{" "}
            <strong className="font-semibold text-[#18241B]">
              {totalTrees.toLocaleString("pt-BR")}
            </strong>{" "}
            árvores ({totalViews.toLocaleString("pt-BR")} fotos)
          </span>
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="inline-flex min-h-[36px] items-center gap-1.5 rounded border border-[#D5E0D1] bg-white px-2.5 py-1.5 text-xs font-medium text-[#2D4234] transition-colors hover:border-[#2D6A4F] hover:text-[#1B4332]"
          >
            <span>{APP_LABELS.apiDocs}</span>
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </header>
  );
}
