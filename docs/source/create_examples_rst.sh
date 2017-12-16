ipynb_names=(model_search guru_example)
rst_names=(train_example guru_example)

for i in 0 1
do
ipython nbconvert ../../example/${ipynb_names[$i]}.ipynb --to rst
mv ../../example/${ipynb_names[$i]}.rst ./${rst_names[$i]}.rst
mv ../../example/${ipynb_names[$i]}_files/* ./images
rm -d ../../example/${ipynb_names[$i]}_files
sed -i -- "s/${ipynb_names[$i]}_files/images/g" ${rst_names[$i]}.rst
done
