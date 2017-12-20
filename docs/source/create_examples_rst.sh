echo -e "Deleting old images"
rm images/*

for file in ../../example/*.ipynb
do
    name_with_path=${file%.*}
    name=${file##*/}
    name=${name%.*}

    echo -e "\nMaking rst from ${name}\n"

    jupyter nbconvert $file --to rst
    mv $name_with_path.rst ./$name.rst
    mv ${name_with_path}_files/* ./images
    rm -d ${name_with_path}_files
    sed -i -- "s/${name}_files/images/g" ${name}.rst
done
