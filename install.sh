apt-get update
apt-get -y upgrade
apt-get -y install vim
apt-get -y install apt-utils
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install -y -c rdkit -c mordred-descriptor mordred=1.1.2
conda install -y -c conda-forge pybel
conda install -y -c openbabel openbabel=3.0.0
conda install -y -c conda-forge tqdm=4.54.1
pip install --upgrade "pip < 21.0"
pip install --upgrade setuptools wheel
pip install numpy==1.16.4
pip install grpcio==1.8.6
pip install tensorflow==1.13.1
pip install pandas==0.24.2
pip install scikit-learn==0.20.2
pip install scipy==1.2.1
pip install git+https://github.com/gadsbyfly/PyBioMed
