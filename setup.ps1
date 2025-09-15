# Chemistry Book AI Setup Commands

# Create and activate conda environment
conda env create -f environment.yml
conda activate chemistry-book-ai

# Install package in development mode
pip install -e .

# Verify GPU memory configuration
python -c "from src.utils.gpu_validator import check_gpu_memory; check_gpu_memory()"