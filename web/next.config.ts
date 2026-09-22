import type { NextConfig } from "next";

// A static export, per section 12 of the build prompt: this app is a shell
// on GitHub Pages with no server of its own, making live client side calls
// to the Fly hosted API at build time's NEXT_PUBLIC_API_BASE_URL. images
// unoptimized because Next's built in image pipeline needs a running server
// this deployment never has. trailingSlash so a static host resolves
// /chat to /chat/index.html the same way it resolves any other folder.
//
// basePath is empty for local development (npm run dev, npm run build for
// a manual check) and set to /groundwork by .github/workflows/pages.yml
// (build order step 25) for the real GitHub Pages build, since a project
// page is served from a subpath, not the domain root. Reading it from an
// environment variable rather than hardcoding it here keeps this file
// correct in both contexts without an if statement keyed on NODE_ENV.
const basePath = process.env.NEXT_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  basePath,
  assetPrefix: basePath.length > 0 ? basePath : undefined,
};

export default nextConfig;
