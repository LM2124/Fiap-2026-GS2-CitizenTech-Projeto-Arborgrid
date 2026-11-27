import { useEffect, useState } from "react";

/** Telas abaixo do breakpoint `lg` do Tailwind, onde as colunas viram uma pilha só. */
export const NARROW_SCREEN_QUERY = "(max-width: 1023.98px)";
/** Aparelhos de toque (celular, tablet), onde a câmera faz sentido e arrastar arquivo não. */
export const TOUCH_DEVICE_QUERY = "(pointer: coarse)";

function matches(query: string): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(query).matches
    : false;
}

export function useMediaQuery(query: string): boolean {
  const [isMatch, setIsMatch] = useState(() => matches(query));
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia(query);
    const onChange = () => setIsMatch(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);
  return isMatch;
}

/**
 * Em tela estreita, rola a página até o elemento: no celular, a foto e o resultado ficam
 * abaixo dos controles, e sem isso a mudança acontece fora da tela.
 */
export function revealOnNarrowScreen(
  el: Element | null,
  block: ScrollLogicalPosition = "start",
): void {
  if (!el || !matches(NARROW_SCREEN_QUERY)) return;
  const reduceMotion = matches("(prefers-reduced-motion: reduce)");
  el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block });
}
