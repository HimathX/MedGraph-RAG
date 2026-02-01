
Write-Host "🐳 Building Docker image..."
docker build -t medgraph-rag .

if ($LASTEXITCODE -eq 0) {
    Write-Host "🚀 Running Docker container..."
    Write-Host "📂 Reading environment variables from .env"
    
    # Run with --env-file which automatically loads VAR=VAL from the file
    docker run -p 8501:8501 --env-file .env medgraph-rag
} else {
    Write-Host "❌ Build failed"
}
