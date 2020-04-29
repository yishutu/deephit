conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install -y -c rdkit -c mordred-descriptor mordred
conda install -y -c conda-forge pybel
conda install -y -c openbabel openbabel
pip install numpy==1.16.4
pip install tensorflow==1.13.1
pip install pandas==0.24.2
pip install scikit-learn==0.20.2
pip install scipy==1.2.1
pip install git+https://github.com/gadsbyfly/PyBioMed
