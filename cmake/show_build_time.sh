#
# Show time taken to build a target 
#
# Input:
# $1 : file containing start of build date in ms from epoch (from cmd "date +%s%3N" with add_custom_command(... POST_BUILD ...))
# $2 : file containing end of build date in ms from epoch (from cmd "date +%s%3N" with add_custom_command(... POST_BUILD ...))
# $3 : name of target built
#
#



start="$(cat $1)"
end="$(cat $2)"

target="$3"
# echo "${end} - ${start}"


time=$(expr ${end} - ${start})


echo "Building target ${target} took ${time}ms."
