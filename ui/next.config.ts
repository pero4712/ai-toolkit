import type { NextConfig } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';

const versionFile = readFileSync(join(__dirname, '..', 'version.py'), 'utf8');
const versionMatch = versionFile.match(/VERSION\s*=\s*["']([^"']+)["']/);
const appVersion = versionMatch ? versionMatch[1] : 'unknown';

// git hash baked in at build time; the docker image builds from a real clone
// so this resolves there too. Falls back gracefully when git is unavailable.
let gitCommit = '';
try {
  gitCommit = execSync('git rev-parse --short HEAD', { cwd: __dirname, stdio: ['ignore', 'pipe', 'ignore'] })
    .toString()
    .trim();
} catch {
  gitCommit = '';
}

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_APP_VERSION: appVersion,
    NEXT_PUBLIC_GIT_COMMIT: gitCommit,
  },
  // unzipper: its optional S3 integration references @aws-sdk/client-s3, which
  // webpack would otherwise try (and fail) to bundle
  serverExternalPackages: ['macstats', 'osx-temperature-sensor', 'unzipper'],
  async rewrites() {
    return [
      {
        source: '/proxy-8866/:path*',
        destination: 'http://localhost:8866/:path*',
      },
    ];
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push('osx-temperature-sensor', 'macstats');
    }
    return config;
  },
  devIndicators: false,
  typescript: {
    // Remove this. Build fails because of route types
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100gb',
    },
    middlewareClientMaxBodySize: '100gb',
  },
};

export default nextConfig;
