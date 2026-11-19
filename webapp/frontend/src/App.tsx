import { useCallback, useEffect, useState } from "react";
import { AlertCircle, ExternalLink, RefreshCw } from "lucide-react";
import { Navbar, type AppTab } from "./components/Navbar";
import {
  fetchBootstrap,
  formatFriendlyError,
  seedGoldDemo,
} from "./services/api";
import type { MapBootstrapPayload } from "./types/arborgrid";
import { APP_LABELS } from "./utils/labels";
import { CopilotView } from "./views/CopilotView";
import { HomeBIView } from "./views/HomeBIView";
import { OperationalMapView } from "./views/OperationalMapView";
import { VisionLabView } from "./views/VisionLabView";

const TAB_TITLES: Record<AppTab, string> = {
  bi: "Visão Geral | ArborGrid SP",
  mapa: "Mapa de Vistorias | ArborGrid SP",
  visao: "Análise de Imagem | ArborGrid SP",
  assistente: "Assistente | ArborGrid SP",
};

function readInitialTab(): AppTab {
  const path = window.location.pathname.toLowerCase();
  if (path.startsWith("/mapa")) return "mapa";
  if (path.startsWith("/visao")) return "visao";
  if (path.startsWith("/assistente") || path.startsWith("/copiloto")) return "assistente";
  return "bi";
}

function readInitialQueryParams(): { treeId: string | null; regionKey: string | null; prompt: string | null } {
  const params = new URLSearchParams(window.location.search);
  const treeId = params.get("tree_id");
  const regionKey = params.get("region");
  const customPrompt = params.get("prompt");
  return {
    treeId: treeId || null,
    regionKey: regionKey || null,
    prompt:
      customPrompt ||
      (treeId
        ? `Qual o risco e o encaminhamento oficial indicado para a árvore cadastrada #${treeId}?`
        : null),
  };
}

export function App() {
  const initialQuery = readInitialQueryParams();
  const [activeTab, setActiveTab] = useState<AppTab>(readInitialTab);
  const [data, setData] = useState<MapBootstrapPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [focusTreeId, setFocusTreeId] = useState<string | null>(initialQuery.treeId);
  const [focusRegionKey, setFocusRegionKey] = useState<string | null>(initialQuery.regionKey);
  const [assistantPrompt, setAssistantPrompt] = useState<string | null>(initialQuery.prompt);

  const loadBootstrap = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchBootstrap();
      setData(payload);
    } catch (err) {
      setError(formatFriendlyError(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadBootstrap();
  }, [loadBootstrap]);

  useEffect(() => {
    document.title = TAB_TITLES[activeTab];
  }, [activeTab]);

  useEffect(() => {
    const onPopState = () => {
      const nextTab = readInitialTab();
      setActiveTab((prev) => {
        if (prev === "assistente" && nextTab !== "assistente") {
          setFocusTreeId(null);
          setAssistantPrompt(null);
        }
        return nextTab;
      });
      window.scrollTo(0, 0);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const handleSelectTab = (tab: AppTab) => {
    if (activeTab === "assistente" && tab !== "assistente") {
      setFocusTreeId(null);
      setAssistantPrompt(null);
    }
    setActiveTab(tab);
    const url = tab === "bi" ? "/" : `/${tab}`;
    if (window.location.pathname !== url) {
      window.history.pushState({}, "", url);
    }
    window.scrollTo(0, 0);
  };

  const handleNavigate = (
    tab: AppTab,
    options?: { geosampaId?: string; regionKey?: string; prompt?: string },
  ) => {
    if (options?.geosampaId) {
      setFocusTreeId(options.geosampaId);
      if (tab === "assistente" && !options.prompt) {
        setAssistantPrompt(
          `Analise o risco e indique o canal oficial adequado para a árvore GeoSampa #${options.geosampaId}.`,
        );
      }
    }
    if (options?.regionKey) setFocusRegionKey(options.regionKey);
    if (options?.prompt) setAssistantPrompt(options.prompt);
    handleSelectTab(tab);
  };

  const handleSeedGold = async () => {
    await seedGoldDemo();
    const updated = await fetchBootstrap();
    setData(updated);
  };

  const handleConsultAssistantFromMap = (geosampaId: string) => {
    setFocusTreeId(geosampaId);
    setAssistantPrompt(
      `Analise o risco e indique o canal oficial adequado para a árvore GeoSampa #${geosampaId}.`,
    );
    handleSelectTab("assistente");
  };

  const handleSendFromVision = (geosampaId: string, prompt: string) => {
    setFocusTreeId(geosampaId || null);
    setAssistantPrompt(prompt);
    handleSelectTab("assistente");
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#F6F8F5] text-[#18241B]">
      <a
        href="#conteudo-principal"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-[#1B4332] focus:px-4 focus:py-2 focus:text-sm focus:font-semibold focus:text-white focus:shadow-lg"
      >
        Pular para o conteúdo
      </a>
      <Navbar
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        totalTrees={data?.stats.total_trees ?? 0}
        totalViews={data?.stats.total_views ?? 0}
      />

      <main id="conteudo-principal" tabIndex={-1} className="flex-1 pb-10 focus:outline-none">
        {loading && !data ? (
          <div className="flex min-h-[75vh] flex-col items-center justify-center gap-3 text-sm text-[#4A5D4E]">
            <RefreshCw className="h-5 w-5 animate-spin text-[#2D6A4F]" />
            <span>Carregando dados do levantamento...</span>
          </div>
        ) : error && !data ? (
          <div className="flex min-h-[60vh] items-start justify-center px-4 pt-12">
            <div className="w-full max-w-lg rounded-md border border-[#FECACA] bg-[#FEF2F2] p-6 text-center">
              <AlertCircle className="mx-auto h-7 w-7 text-[#991B1B]" />
              <p className="mt-2 text-sm font-medium text-[#7F1D1D]">{error}</p>
              <button
                type="button"
                onClick={() => void loadBootstrap()}
                className="mt-4 rounded bg-[#1B4332] px-4 py-2 text-xs font-semibold text-white hover:bg-[#143325]"
              >
                Tentar novamente
              </button>
            </div>
          </div>
        ) : data ? (
          <>
            {activeTab === "bi" && (
              <HomeBIView
                data={data}
                onNavigate={handleNavigate}
                onSeedGold={handleSeedGold}
              />
            )}
            {activeTab === "mapa" && (
              <OperationalMapView
                data={data}
                initialTreeId={focusTreeId}
                initialRegionKey={focusRegionKey}
                onConsultAssistant={handleConsultAssistantFromMap}
              />
            )}
            {activeTab === "visao" && (
              <VisionLabView
                trees={data.trees}
                onSendToAssistant={handleSendFromVision}
              />
            )}
            {activeTab === "assistente" && (
              <CopilotView
                trees={data.trees}
                initialTreeId={focusTreeId}
                initialPrompt={assistantPrompt}
                onConsumeInitialPrompt={() => setAssistantPrompt(null)}
                onClearTreeContext={() => setFocusTreeId(null)}
              />
            )}
          </>
        ) : null}
      </main>

      <footer className="border-t border-[#D5E0D1] bg-white py-5 text-xs text-[#4A5D4E]">
        <div className="mx-auto flex max-w-[1680px] flex-wrap items-center justify-between gap-3 px-4 sm:px-8">
          <p>
            <strong className="font-semibold text-[#18241B]">ArborGrid SP</strong>, projeto acadêmico FIAP Global Solution 2026 (CitizenTech).
          </p>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <p>
              Fontes externas: cadastro de árvores da Prefeitura (GeoSampa) e histórico de vistorias (SISGAU), imagens de rua (Google Street View), previsão do tempo (Open-Meteo) e mapas (Esri).
            </p>
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 font-medium text-[#1B4332] hover:underline"
            >
              <span>{APP_LABELS.apiDocs}</span>
              <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
