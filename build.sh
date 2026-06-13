#!/bin/bash
# PhotoFinder Next build script
# Builds the Tauri app and copies dist folder to bundle

set -e

echo "=== Building PhotoFinder Next ==="

# Build the Tauri app
cd src-tauri
cargo tauri build

# Copy dist folder to bundle (required for macOS bundle)
BUNDLE_PATH="target/release/bundle/macos/PhotoFinder Next.app/Contents/Resources"
if [ -d "../dist" ]; then
    echo "Copying dist to bundle..."
    cp -r ../dist "$BUNDLE_PATH/"
else
    echo "Warning: dist folder not found at ../dist"
fi

# Copy to Applications
rm -rf "/Applications/PhotoFinder Next.app" 2>/dev/null || true
cp -r "$BUNDLE_PATH/../.." "/Applications/PhotoFinder Next.app"
echo "Installed to /Applications/PhotoFinder Next.app"