import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

export default defineConfig({
  site: "https://sjq8921.github.io",
  integrations: [
    tailwind({
      applyBaseStyles: false
    })
  ]
});
