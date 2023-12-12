echo "Installing Python Dependencies"
pip install -r ml/requirements.txt --no-cache-dir

echo "Installing ml modules"
pip install -e ml
