

files="$(find)"
files="$(find -not -path "./kernels/*/*")"

#~ echo "$files"
echo $(wc -l $files)
#~ for file in $files
#~ do
#~ l=$(wc -l $file)

#~ done

