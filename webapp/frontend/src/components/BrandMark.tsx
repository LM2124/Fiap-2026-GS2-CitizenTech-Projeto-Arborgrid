interface BrandMarkProps {
  className?: string;
}

/**
 * Marca gráfica do ArborGrid SP (árvore estilizada atravessada por linha de fiação),
 * idêntica ao favicon em public/favicon.svg.
 */
export function BrandMark({ className = "h-8 w-8 shrink-0" }: BrandMarkProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden="true"
      className={className}
    >
      <rect width="64" height="64" rx="14" fill="#1B4332" />
      <path
        d="M32 13C23 13 17 19.8 17 28C17 34.6 21.4 40 27.5 41.7V50H36.5V41.7C42.6 40 47 34.6 47 28C47 19.8 41 13 32 13Z"
        fill="#D8F3DC"
      />
      <path
        d="M32 21V51M32 35L24.5 28.5M32 30.5L39.5 24.5"
        stroke="#1B4332"
        strokeWidth="2.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M10 22Q32 26.5 54 22"
        stroke="#F59E0B"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <circle cx="10" cy="22" r="2.5" fill="#F59E0B" />
      <circle cx="54" cy="22" r="2.5" fill="#F59E0B" />
    </svg>
  );
}
