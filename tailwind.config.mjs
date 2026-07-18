/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Microsoft YaHei",
          "Arial",
          "sans-serif"
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Consolas", "monospace"]
      },
      boxShadow: {
        card: "0 18px 50px rgba(0, 0, 0, 0.06)",
        soft: "0 8px 26px rgba(0, 0, 0, 0.045)"
      }
    }
  },
  plugins: []
};
