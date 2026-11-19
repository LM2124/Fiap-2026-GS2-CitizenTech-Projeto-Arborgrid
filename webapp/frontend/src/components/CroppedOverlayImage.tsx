interface CroppedOverlayImageProps {
  src: string;
  alt: string;
  className?: string;
}

/**
 * Exibe imagem de overlay recortando via CSS a tarja de debug (~29 px de 640 px, ~4,8%)
 * queimada no topo pelo pipeline, preservando a proporção da cena.
 */
export function CroppedOverlayImage({ src, alt, className = "" }: CroppedOverlayImageProps) {
  return (
    <div
      className={`aspect-[640/609] w-full overflow-hidden rounded border border-[#D5E0D1] bg-[#F6F8F5] ${className}`}
    >
      <img
        src={src}
        alt={alt}
        loading="lazy"
        className="-mt-[4.85%] block h-auto w-full select-none"
      />
    </div>
  );
}
