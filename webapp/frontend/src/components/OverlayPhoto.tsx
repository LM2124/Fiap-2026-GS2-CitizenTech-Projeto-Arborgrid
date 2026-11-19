import { useCallback, useEffect, useState } from "react";
import { Eye, EyeOff } from "lucide-react";

interface OverlayPhotoProps {
  overlayUrl?: string | null;
  imageUrl?: string | null;
  overlayAlt: string;
  originalAlt: string;
  className?: string;
}

/**
 * Exibe foto quadrada (aspect-square object-cover) sem recorte de tarja.
 * Remonta o <img> pela URL para evitar imagem fantasma na troca de árvores,
 * pré-carrega a foto original e alterna para ela em hover, foco ou botão em telas sem hover.
 */
export function OverlayPhoto({
  overlayUrl,
  imageUrl,
  overlayAlt,
  originalAlt,
  className = "",
}: OverlayPhotoProps) {
  const hasOverlay = Boolean(overlayUrl);
  const hasOriginal = Boolean(imageUrl) && imageUrl !== overlayUrl;
  const canToggle = hasOverlay && hasOriginal;

  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [manualShowOriginal, setManualShowOriginal] = useState(false);
  const [loadedUrls, setLoadedUrls] = useState<Record<string, boolean>>({});
  const [erroredUrls, setErroredUrls] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setIsHovered(false);
    setIsFocused(false);
    setManualShowOriginal(false);
  }, [overlayUrl, imageUrl]);

  // Pré-carrega a foto original quando o componente monta para troca instantânea
  useEffect(() => {
    if (!imageUrl || imageUrl === overlayUrl) return;
    let cancelled = false;
    const preloader = new Image();
    preloader.onload = () => {
      if (!cancelled) {
        setLoadedUrls((prev) => (prev[imageUrl] ? prev : { ...prev, [imageUrl]: true }));
      }
    };
    preloader.src = imageUrl;
    if (preloader.complete && preloader.naturalWidth > 0) {
      setLoadedUrls((prev) => (prev[imageUrl] ? prev : { ...prev, [imageUrl]: true }));
    }
    return () => {
      cancelled = true;
    };
  }, [imageUrl, overlayUrl]);

  const showOriginal = canToggle
    ? manualShowOriginal || isHovered || isFocused
    : !hasOverlay && Boolean(imageUrl);

  const displaySrc = showOriginal ? (imageUrl ?? "") : (overlayUrl ?? imageUrl ?? "");
  const displayAlt = showOriginal ? originalAlt : overlayAlt;
  const isLoaded = Boolean(displaySrc && loadedUrls[displaySrc]);
  const isErrored = Boolean(displaySrc && erroredUrls[displaySrc]);

  const handleImgRef = useCallback(
    (el: HTMLImageElement | null) => {
      if (el && el.complete && el.naturalWidth > 0 && displaySrc) {
        setLoadedUrls((prev) => (prev[displaySrc] ? prev : { ...prev, [displaySrc]: true }));
      }
    },
    [displaySrc],
  );

  const markLoaded = (url: string) => {
    setLoadedUrls((prev) => (prev[url] ? prev : { ...prev, [url]: true }));
  };

  const markErrored = (url: string) => {
    setErroredUrls((prev) => (prev[url] ? prev : { ...prev, [url]: true }));
  };

  if (!displaySrc) {
    return (
      <div
        className={`flex aspect-square w-full items-center justify-center rounded border border-[#D5E0D1] bg-[#EEF2EC] text-xs text-[#4A5D4E] ${className}`}
      >
        Foto indisponível
      </div>
    );
  }

  return (
    <div className={className}>
      <div
        tabIndex={canToggle ? 0 : undefined}
        onMouseEnter={canToggle ? () => setIsHovered(true) : undefined}
        onMouseLeave={canToggle ? () => setIsHovered(false) : undefined}
        onFocus={canToggle ? () => setIsFocused(true) : undefined}
        onBlur={canToggle ? () => setIsFocused(false) : undefined}
        className={`keep-colors relative aspect-square w-full overflow-hidden rounded border border-[#D5E0D1] bg-[#EEF2EC] ${
          canToggle
            ? "cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2D6A4F]"
            : ""
        }`}
      >
        {!isLoaded && !isErrored && (
          <div
            aria-hidden="true"
            className="absolute inset-0 bg-[#E6ECE3] motion-safe:animate-pulse"
          />
        )}

        {isErrored ? (
          <div className="flex h-full w-full items-center justify-center bg-[#EEF2EC] px-3 text-center text-xs text-[#4A5D4E]">
            Não foi possível carregar esta foto.
          </div>
        ) : (
          <img
            key={displaySrc}
            ref={handleImgRef}
            src={displaySrc}
            alt={displayAlt}
            onLoad={() => markLoaded(displaySrc)}
            onError={() => markErrored(displaySrc)}
            className={`keep-colors block h-full w-full select-none object-cover transition-opacity duration-150 ${
              isLoaded ? "opacity-100" : "opacity-0"
            }`}
          />
        )}

        {canToggle && showOriginal && isLoaded && (
          <span className="pointer-events-none absolute left-2 top-2 rounded bg-[#18241B]/80 px-2 py-0.5 text-[11px] font-medium text-white">
            Foto original
          </span>
        )}
      </div>

      {canToggle && (
        <button
          type="button"
          aria-pressed={manualShowOriginal}
          onClick={() => setManualShowOriginal((prev) => !prev)}
          className="mt-1.5 hidden min-h-[28px] items-center gap-1 rounded border border-[#D5E0D1] bg-white px-2 py-0.5 text-[11px] font-medium text-[#1B4332] transition-colors hover:border-[#2D6A4F] hover:bg-[#EEF4EC] [@media(hover:none)]:inline-flex"
        >
          {manualShowOriginal ? (
            <>
              <Eye className="h-3 w-3 shrink-0" aria-hidden="true" />
              <span>Ver marcações</span>
            </>
          ) : (
            <>
              <EyeOff className="h-3 w-3 shrink-0" aria-hidden="true" />
              <span>Ver foto sem marcações</span>
            </>
          )}
        </button>
      )}
    </div>
  );
}
