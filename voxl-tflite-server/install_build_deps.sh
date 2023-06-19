#!/bin/bash
################################################################################
# Copyright (c) 2022 ModalAI, Inc. All rights reserved.
################################################################################

# Script to install build dependencies in voxl-cross docker image

# list all your dependencies here. Note for packages that have AMD64 equivalents
# in the ubuntu repositories you should specify the arm64 architecture to make
# sure the correct one is installed in voxl-cross.
DEPS_QRB5165="
libmodal-pipe
libmodal-json
voxl-opencv
qrb5165-tflite"

DEPS_APQ8096="
libmodal-pipe
libmodal-json
voxl-opencv
apq8096-tflite"

## this list is just for tab-completion
## it's not an exhaustive list of platforms available.
AVAILABLE_PLATFORMS="qrb5165 apq8096"


print_usage(){
	echo ""
	echo " Install build dependencies from a specified repository."
	echo " For apq8096 \"dev\" and \"stable\" repos the packages"
	echo " will be pulled as IPKs and installed with opkg."
	echo " Otherwise debs will be pulled with apt."
	echo ""
	echo " Usage:"
	echo "  ./install_build_deps.sh {platform} {section}"
	echo ""
	echo " Examples:"
	echo ""
	echo "  ./install_build_deps.sh qrb5165 dev"
	echo "        Install from qrb5165 development repo."
	echo ""
	echo "  ./install_build_deps.sh qrb5165 sdk-1.0"
	echo "        Install from qrb5165 sdk-1.0 repo."
	echo ""
	echo "  ./install_build_deps.sh apq8096 dev"
	echo "        Install from apq8096 development repo."
	echo ""
	echo "  ./install_build_deps.sh apq8096 staging"
	echo "        Install from apq8096 staging repo."
	echo ""
	echo ""
	echo " These examples are not an exhaustive list."
	echo " Any platform and section in this deb repo can be used:"
	echo "     http://voxl-packages.modalai.com/dists/"
	echo ""
}

# make sure two arguments were given
if [ "$#" -ne 2 ]; then
	print_usage
	exit 1
fi
MODE="DEB"

## convert arguments to lower case for robustness
PLATFORM=$(echo "$1" | tr '[:upper:]' '[:lower:]')
SECTION=$(echo "$2" | tr '[:upper:]' '[:lower:]')

# use ipk mode for apq8096
if [ "$PLATFORM" == "apq8096" ]; then
	MODE="IPK"
fi


# install deb packages with apt
if [ "$MODE" == "DEB" ]; then
	echo "using $PLATFORM $SECTION debian repo"
	# write in the new entry
	DPKG_FILE="/etc/apt/sources.list.d/modalai.list"
	LINE="deb [trusted=yes] http://voxl-packages.modalai.com/ ./dists/$PLATFORM/$SECTION/binary-arm64/"
	sudo echo "${LINE}" > ${DPKG_FILE}

	## make sure we have the latest package index
	## only pull from voxl-packages to save time
	sudo apt-get update -o Dir::Etc::sourcelist="sources.list.d/modalai.list" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"

	## install the user's list of dependencies
	echo "installing: $DEPS_QRB5165"
	sudo apt install -y $DEPS_QRB5165

# install IPK packages with opkg
else
	echo "using $PLATFORM $SECTION repo"
	OPKG_CONF="/etc/opkg/opkg.conf"
	# delete any existing repository entries
	sudo sed -i '/voxl-packages.modalai.com/d' ${OPKG_CONF}

	# add arm64 architecture if necessary
	if ! grep -q "arch arm64" "${OPKG_CONF}"; then
		echo "adding arm64 to opkg conf"
		sudo echo "arch arm64 7" >> ${OPKG_CONF}
	fi

	# write in the new entry
	LINE="src/gz ${SECTION} http://voxl-packages.modalai.com/dists/$PLATFORM/${SECTION}/binary-arm64/"
	sudo echo "$LINE" >> ${OPKG_CONF}
	sudo echo "" >> ${OPKG_CONF}

	## make sure we have the latest package index
	sudo opkg update

	echo "installing: $DEPS_APQ8096"

	# install/update each dependency
	for i in ${DEPS_APQ8096}; do
		# this will also update if already installed!
		sudo opkg install --nodeps $i
	done

fi

echo ""
echo "Done installing dependencies"
echo ""
exit 0
