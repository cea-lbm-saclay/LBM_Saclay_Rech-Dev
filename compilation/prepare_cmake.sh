#!/bin/bash


source $LBM_SACLAY_DIR/compilation/pprint.sh

cd $LBM_SACLAY_DIR


# get list of problems
PROBLEMS_DIR=$LBM_SACLAY_DIR/src/kernels
PROBLEM_LIST=($(cd $PROBLEMS_DIR && ls -d */ | sed 's:/*$::'))

# Show the list of problems found, with corresponding numbers for choosing
echo "The following problems are currently implemented:"
for i in ${!PROBLEM_LIST[@]}
do
echo "$(printf " %-2d" $i) ${PROBLEM_LIST[$i]}"
done

# Ask for the problems to prepare
echo "Choose which problems to include by indicating a list of space or comma separated numbers, eg '0 1' or '0,1'."
echo "Write 'all' to include all problems."
read -p "Problem numbers: " PBM_NUMBERS

PBM_NUMBERS=${PBM_NUMBERS//","/" "}


if [ "${PBM_NUMBERS}" = "all" ]
then
    PBM_NUMBERS=${!PROBLEM_LIST[@]}
    BUILD_ALL="true"
else
    BUILD_ALL="false"
fi

for PBM_NUMBER in $PBM_NUMBERS
do
    NUMBER=${PBM_NUMBER:-0}
    
    NOT_IN_THE_LIST="true"
    for item in ${!PROBLEM_LIST[@]}
    do
        if [ ${NUMBER} = ${item} ];
        then
            NOT_IN_THE_LIST="false"
            #echo "${NUMBER} is a valid problem number"
        fi
    done
    if ( $NOT_IN_THE_LIST ); then
        echo "ERROR: ${PBM_NUMBER} is not a valid problem number"
        exit 1
    fi

    PBM_NAME="${PROBLEM_LIST[$PBM_NUMBER]}"
    echo "Problem n°$(printf "%-2d" $NUMBER) added to compile list (${PBM_NAME})"
    PBM_TO_PREPARE="$PBM_TO_PREPARE $PBM_NAME"
    PBM_COMPILE_LIST="${PBM_COMPILE_LIST};${PBM_NAME}"
    
    if ( $BUILD_ALL )
    then
        PBM_BUILD_NAME="_all"
    else
        PBM_BUILD_NAME="${PBM_BUILD_NAME}_${PBM_NAME}"
    fi

done
PBM_COMPILE_LIST="${PBM_COMPILE_LIST:1}"
PBM_BUILD_NAME="${PBM_BUILD_NAME:1}"

# Create build dir for this architecture
cd $LBM_SACLAY_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR


# Create build dir for this combination of problems and run cmake
cd $BUILD_DIR
mkdir -p "build_${PBM_BUILD_NAME}"
cd "build_${PBM_BUILD_NAME}"

echo "//==================================================="
CMAKE_CMD="cmake ${CMAKE_OPTIONS} -DPROBLEM=${PBM_COMPILE_LIST} ${LBM_SACLAY_DIR}"
echo "cmake command is : ${CMAKE_CMD}"
cmake ${CMAKE_OPTIONS} -DPROBLEM=${PBM_COMPILE_LIST} ${LBM_SACLAY_DIR} > /dev/null

echo "//==================================================="
echo ""
echo "build configured in:"
pprint "${BUILD_DIR}/build_${PBM_BUILD_NAME}" "BoldCyan"
echo "go there and use make to compile"


