cd "$(dirname "$0")"/..
wget https://zenodo.org/api/records/15676408/files/data.tar.gz/content
tar -xvzf content
rm content