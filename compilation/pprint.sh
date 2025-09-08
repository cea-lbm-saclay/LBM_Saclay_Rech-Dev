
function pprint() {

    text=$1
    style=$2
    newLine=${3:-true}
    format="%s"
    case "$style" in
        'success')format="\033[0;32m%s\033[0m";;
        'error')format="\033[1;31m%s\033[0m";;
        'info')format="\033[33;33m%s\033[0m";;

        'Black')format="\033[0;30m%s\033[0m";;
        'Red')format="\033[0;31m%s\033[0m";;
        'Green')format="\033[0;32m%s\033[0m";;
        'Yellow')format="\033[0;33m%s\033[0m";;
        'Blue')format="\033[0;34m%s\033[0m";;
        'Purple')format="\033[0;35m%s\033[0m";;
        'Cyan')format="\033[0;36m%s\033[0m";;
        'Gray')format="\033[0;37m%s\033[0m";;
        'Graphite')format="\033[1;30m%s\033[0m";;
        
        'BoldRed')format="\033[1;31m%s\033[0m";;
        'BoldGreen')format="\033[1;32m%s\033[0m";;
        'BoldYellow')format="\033[1;33m%s\033[0m";;
        'BoldBlue')format="\033[1;34m%s\033[0m";;
        'BoldPurple')format="\033[1;35m%s\033[0m";;
        'BoldCyan')format="\033[1;36m%s\033[0m";;
        'BoldWhite')format="\033[1;37m%s\033[0m";;
        
        'BgBlack')format="\033[40;1;37m%s\033[0m";;
        'BgRed')format="\033[41;1;37m%s\033[0m";;
        'BgGreen')format="\033[42;1;37m%s\033[0m";;
        'BgYellow')format="\033[43;1;37m%s\033[0m";;
        'BgBlue')format="\033[44;1;37m%s\033[0m";;
        'BgPurple')format="\033[45;1;37m%s\033[0m";;
        'BgCyan')format="\033[46;1;37m%s\033[0m";;
        'BgGray')format="\033[47;1;37m%s\033[0m";;
        
        'Underscore')format="\033[4;37m%s\033[0m";;
        'Inverted')format="\033[7;37m%s\033[0m";;
        'Blink')format="\033[5;37m%s\033[0m";;
    esac
    
    if ( $newLine )
    then
        format="${format}\n"
    fi

    printf $format $text
}
