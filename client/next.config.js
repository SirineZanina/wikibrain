/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Needed for the slim Docker runtime stage.
  output: "standalone",
};

module.exports = nextConfig;
