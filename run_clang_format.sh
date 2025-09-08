
if ! command -v clang-format &> /dev/null
then
    echo "clang-format could not be found"
    exit 1
fi
echo $(clang-format --version)
echo "Version 13 needed at minimum" # TODO: implement automatic check

CLANG_FORMAT_OPT="--style=file -Werror"

# list project files
SOURCES=$(find src -name *.cpp)
HEADERS=$(find src -name *.h)
FILES="${SOURCES} ${HEADERS}"

FORMATTED=true
COUNT=0
for FILE in $FILES
do
    if ( ! $(clang-format ${CLANG_FORMAT_OPT} --dry-run $FILE > /dev/null 2>&1) )
    then
        FORMATTED=false
        NON_FORMATTED_FILES="${NON_FORMATTED_FILES} $FILE"
        COUNT=$((COUNT+1))
    fi
done


if ( ! $($FORMATTED) )
then
    echo "There are ${COUNT} non formatted files."
    echo "Run this script with option -l to see the list."
    echo "Run this script with option -f to apply formatting."
else
    echo "Nothing to be done, source code is correctly formatted."
fi

while getopts 'flh' opt; do
    case "$opt" in
    f)
        echo "Formatting source files"
        for FILE in $FILES
        do
            clang-format ${CLANG_FORMAT_OPT} -i $FILE 
        done
    ;;
    l)
        echo "Listing files that do not comply to formatting rules:"
        for FILE in $NON_FORMATTED_FILES
        do
            echo "${FILE}"
        done
    ;;

    ?|h)
        echo "Usage: $(basename $0) [-f] [-l]"
        echo "    -l : list files needing formatting"
        exit 1
    ;;
    esac
done
shift "$(($OPTIND -1))"






if ( ! $($FORMATTED) )
then
    exit 1
fi


