mkdir -p datasets

cd datasets

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
tar -xzvf gdb9.tar.gz
mv gdb9.sdf qm9.sdf
mv gdb9.sdf.csv qm9.sdf.csv
rm gdb9.tar.gz

cd ..

mkdir -p figures
