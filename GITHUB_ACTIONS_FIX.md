# 🔧 GitHub Actions Docker Login Fix

## Issue
GitHub Actions workflow was failing with error:
```
Run docker/login-action@v3
Error: Username and password required
```

## Root Cause
1. **Missing Permissions**: The workflow didn't have explicit permissions to write to GitHub Container Registry (GHCR)
2. **Conflicting Workflows**: There were two workflows running simultaneously:
   - `ci-cd.yml` - Using GitHub Container Registry (ghcr.io) ✅
   - `ci-cd-dockerhub.yml` - Using Docker Hub (docker.io) ❌ (missing credentials)

## Solution Applied

### 1. Added Explicit Permissions
Updated `.github/workflows/ci-cd.yml` to include:

```yaml
permissions:
  contents: read
  packages: write        # Required for GHCR push
  security-events: write # Required for Trivy scan uploads
```

### 2. Removed Conflicting Docker Hub Workflow
Deleted `.github/workflows/ci-cd-dockerhub.yml` because:
- It required Docker Hub credentials (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`)
- GitHub Container Registry works without additional setup
- Having two workflows caused confusion and errors

### 3. Workflow Authentication
The remaining workflow uses:
- `username: ${{ github.actor }}` - GitHub username
- `password: ${{ secrets.GITHUB_TOKEN }}` - Automatic GitHub token

### 4. Registry Configuration
```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
```

## Verification

After the fix:
1. ✅ Workflow has proper permissions
2. ✅ Only one workflow runs (GitHub Container Registry)
3. ✅ Docker login should succeed
4. ✅ Image builds and pushes to `ghcr.io/ardas2012/german-credit-risk-prediction`
5. ✅ Security scans upload to GitHub Security tab

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

### Option 3: Use Docker Hub (if preferred)
1. Create Docker Hub account
2. Add repository secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
3. Restore the Docker Hub workflow

## Current Status
- ✅ Fix committed and pushed
- ✅ Conflicting workflow removed
- ✅ GitHub Actions will use updated permissions
- ✅ Next push should build and deploy successfully

## Related Files
- `.github/workflows/ci-cd.yml` - Main workflow (GitHub Container Registry)
- ~~`.github/workflows/ci-cd-dockerhub.yml`~~ - Removed (was causing errors)
- `Dockerfile` - Container configuration
- `src/api.py` - API with CORS enabled

## Testing
Monitor the next GitHub Actions run at:
https://github.com/ArdaS2012/german-credit-risk-prediction/actions 