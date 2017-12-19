for file in ../../example/*.ipynb
do
    echo "making rst from $file"
    name_with_path=${file%.*}
    name=${file##*/}
    name=${name%.*}
    ipython nbconvert $file --to rst
    mv $name_with_path.rst ./$name.rst
    mv ${name_with_path}_files/* ./images
    rm -d ${name_with_path}_files
    sed -i -- "s/${name}_files/images/g" ${name}.rst
done
