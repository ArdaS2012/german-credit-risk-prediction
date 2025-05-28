# 🔧 GitHub Actions Docker Login Fix

## Issue
GitHub Actions workflow was failing with error:
```
Run docker/login-action@v3
Error: Username and password required
```

## Root Cause
The workflow didn't have explicit permissions to write to GitHub Container Registry (GHCR).

## Solution Applied

### 1. Added Explicit Permissions
Updated `.github/workflows/ci-cd.yml` to include:

```yaml
permissions:
  contents: read
  packages: write        # Required for GHCR push
  security-events: write # Required for Trivy scan uploads
```

### 2. Workflow Authentication
The workflow uses:
- `username: ${{ github.actor }}` - GitHub username
- `password: ${{ secrets.GITHUB_TOKEN }}` - Automatic GitHub token

### 3. Registry Configuration
```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
```

## Verification

After the fix:
1. ✅ Workflow has proper permissions
2. ✅ Docker login should succeed
3. ✅ Image builds and pushes to `ghcr.io/ardas2012/german-credit-risk-prediction`
4. ✅ Security scans upload to GitHub Security tab

## Alternative Solutions

If issues persist, you can also:

### Option 1: Personal Access Token
1. Create PAT with `write:packages` scope
2. Add as repository secret `DOCKER_TOKEN`
3. Update workflow:
   ```yaml
   password: ${{ secrets.DOCKER_TOKEN }}
   ```

### Option 2: Repository Settings
1. Go to repository Settings → Actions → General
2. Set "Workflow permissions" to "Read and write permissions"
3. Check "Allow GitHub Actions to create and approve pull requests"

## Current Status
- ✅ Fix committed and pushed
- ✅ GitHub Actions will use updated permissions
- ✅ Next push should build and deploy successfully

## Related Files
- `.github/workflows/ci-cd.yml` - Updated workflow
- `Dockerfile` - Container configuration
- `src/api.py` - API with CORS enabled

## Testing
Monitor the next GitHub Actions run at:
https://github.com/ArdaS2012/german-credit-risk-prediction/actions 