# GitHub Container Registry Setup Guide

## Problem
You're encountering this error when pushing Docker images:
```
ERROR: denied: installation not allowed to Create organization package
```

## Solutions

### Solution 1: Enable GitHub Container Registry Permissions (Recommended)

1. **Go to your repository settings:**
   - Navigate to: https://github.com/ArdaS2012/german-credit-risk-prediction/settings
   - Click on "Actions" in the left sidebar
   - Click on "General"

2. **Update Workflow Permissions:**
   - Scroll down to "Workflow permissions"
   - Select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"
   - Click "Save"

3. **Enable Package Creation:**
   - Go to your GitHub profile settings: https://github.com/settings/packages
   - Under "Package creation", make sure "Public" is enabled
   - Or go to your organization settings if this is an organization repo

### Solution 2: Create a Personal Access Token

If Solution 1 doesn't work, create a PAT with package permissions:

1. **Create Personal Access Token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes:
     - `write:packages`
     - `read:packages`
     - `delete:packages` (optional)

2. **Add Token to Repository Secrets:**
   - Go to: https://github.com/ArdaS2012/german-credit-risk-prediction/settings/secrets/actions
   - Click "New repository secret"
   - Name: `GHCR_TOKEN`
   - Value: Your PAT token

3. **Update CI/CD workflow** (see updated .github/workflows/ci-cd.yml)

### Solution 3: Use Docker Hub Instead

If GitHub Container Registry continues to have issues, switch to Docker Hub:

1. **Create Docker Hub account** at https://hub.docker.com
2. **Create repository secret:**
   - Name: `DOCKERHUB_USERNAME`
   - Value: Your Docker Hub username
   - Name: `DOCKERHUB_TOKEN`
   - Value: Your Docker Hub access token

### Solution 4: Build Locally Without Registry

For testing purposes, you can build the Docker image locally:

```bash
# Build the image locally
docker build -t german-credit-risk-prediction:latest .

# Run the container
docker run -p 8000:8000 german-credit-risk-prediction:latest
```

## Testing the Fix

After implementing Solution 1 or 2:

1. **Push a small change to trigger the workflow:**
   ```bash
   git add .
   git commit -m "test: trigger CI/CD pipeline"
   git push origin main
   ```

2. **Monitor the GitHub Actions tab:**
   - Go to: https://github.com/ArdaS2012/german-credit-risk-prediction/actions
   - Watch the build job to see if it succeeds

## Troubleshooting

If you still get permission errors:

1. **Check if the repository is private:**
   - Private repos might have different permission requirements
   - Consider making it public for testing

2. **Verify your GitHub username:**
   - Make sure `ArdaS2012` is correct
   - Check if it's a personal account vs organization

3. **Check GitHub Container Registry status:**
   - Visit: https://www.githubstatus.com/
   - Look for any ongoing issues with Container Registry

## Quick Fix for Immediate Testing

If you need to test the application immediately without fixing registry issues:

```bash
# Build and run locally
cd /home/arda/Schreibtisch/test_interview_project
docker build -t credit-risk-api:local .
docker run -p 8000:8000 credit-risk-api:local
```

Then visit http://localhost:8000/docs to test the API. 