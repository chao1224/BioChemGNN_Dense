wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/sider.csv.gz

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/muv.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/HIV.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pcba.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv
wget -O malaria-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv
wget -O cep-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/SAMPL.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/Lipophilicity.csv

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb7.tar.gz
tar -xzvf gdb7.tar.gz

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm8.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz
tar -xzvf gdb8.tar.gz
rm gdb8.tar.gz

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
tar -xzvf gdb9.tar.gz
mv gdb9.sdf qm9.sdf
mv gdb9.sdf.csv qm9.sdf.csv
rm gdb9.tar.gz


echo "Pulling featurized core pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz
echo "Extracting core pdbbind"
tar -zxvf core_grid.tar.gz
echo "Pulling featurized refined pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/refined_grid.tar.gz
echo "Extracting refined pdbbind"
tar -zxvf refined_grid.tar.gz
echo "Pulling featurized full pdbbind dataset from deepchem"
wget -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/full_grid.tar.gz
echo "Extracting full pdbbind"
tar -zxvf full_grid.tar.gz

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz
tar -xzvf
