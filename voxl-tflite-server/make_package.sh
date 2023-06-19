
USETIMESTAMP=false
MAKE_DEB=true
MAKE_IPK=false

print_usage(){
	echo ""
	echo " Package the current project into a deb or ipk package."
	echo " You must run build.sh first to build the binaries"
	echo " if no arguments are given it builds a deb"
	echo ""
	echo " Usage:"
	echo "  ./make_package.sh"
	echo "  ./make_package.sh deb"
	echo "        Build a DEB package"
	echo ""
	echo "  ./make_package.sh ipk"
	echo "        Build an IPK package"
	echo ""
	echo "  ./make_package.sh timestamp"
	echo "  ./make_package.sh deb timestamp"
	echo "        Build a DEB package with the current timestamp as a"
	echo "        suffix in both the package name and deb filename."
	echo "        This is used by CI for development packages."
	echo ""
	echo ""
}

process_argument () {

	if [ "$#" -ne 1 ]; then
		echo "ERROR process_argument expected 1 argument"
		exit 1
	fi

	## convert argument to lower case for robustness
	arg=$(echo "$1" | tr '[:upper:]' '[:lower:]')
	case ${arg} in
		"")
			;;
		"-t"|"timestamp"|"--timestamp")
			echo "using timestamp suffix"
			USETIMESTAMP=true
			;;
		"-d"|"deb"|"debian"|"--deb"|"--debian")
			MAKE_DEB=true
			;;
		"-i"|"ipk"|"opkg"|"--ipk"|"--opkg")
			MAKE_IPK=true
			MAKE_DEB=false
			;;
		*)
			echo "invalid option"
			print_usage
			exit 1
	esac
}


## parse all arguments or run wizard
for var in "$@"
do
	process_argument $var
done


################################################################################
# variables
################################################################################
VERSION=$(cat pkg/control/control | grep "Version" | cut -d' ' -f 2)
PACKAGE=$(cat pkg/control/control | grep "Package" | cut -d' ' -f 2)
IPK_NAME=${PACKAGE}_${VERSION}.ipk


DATA_DIR=pkg/data
CONTROL_DIR=pkg/control
IPK_DIR=pkg/IPK
DEB_DIR=pkg/DEB

echo "Package Name: " $PACKAGE
echo "version Number: " $VERSION

################################################################################
# start with a little cleanup to remove old files
################################################################################
# remove data directory where 'make install' installed to
sudo rm -rf $DATA_DIR
mkdir $DATA_DIR

# remove ipk and deb packaging folders
rm -rf $IPK_DIR
rm -rf $DEB_DIR

# remove old ipk and deb packages
rm -f *.ipk
rm -f *.deb


################################################################################
## install compiled stuff into data directory with 'make install'
## try this for all 3 possible build folders, some packages are multi-arch
## so both 32 and 64 need installing to pkg directory.
################################################################################

DID_BUILD=false

if [[ -d "build" ]]; then
	cd build && sudo make DESTDIR=../${DATA_DIR} PREFIX=/usr install && cd -
	DID_BUILD=true
fi
if [[ -d "build32" ]]; then
	cd build32 && sudo make DESTDIR=../${DATA_DIR} PREFIX=/usr install && cd -
	DID_BUILD=true
fi
if [[ -d "build64" ]]; then
	cd build64 && sudo make DESTDIR=../${DATA_DIR} PREFIX=/usr install && cd -
	DID_BUILD=true
fi

# make sure at least one directory worked
if [ "$DID_BUILD" = false ] && [ -f "build.sh" ]; then
	echo "neither build/ build32/ or build64/ were found"
	exit 1
fi


################################################################################
## install standard stuff common across ModalAI projects if they exist
################################################################################

if [ -d "services" ]; then
	sudo mkdir -p $DATA_DIR/etc/systemd/system/
	sudo cp services/* $DATA_DIR/etc/systemd/system/
fi

if [ -d "scripts" ]; then
	sudo mkdir -p $DATA_DIR/usr/bin/
	sudo chmod +x scripts/*/*
	if [ $MAKE_DEB == false ]; then
		sudo cp scripts/apq8096/* $DATA_DIR/usr/bin/
	fi
	if [ $MAKE_IPK == false ]; then
		sudo cp scripts/qrb5165/* $DATA_DIR/usr/bin/
	fi
fi

if [ -d "bash_completions" ]; then
	sudo mkdir -p $DATA_DIR/usr/share/bash-completion/completions
	sudo cp bash_completions/* $DATA_DIR/usr/share/bash-completion/completions
fi

if [ -d "misc_files" ]; then
		sudo cp -R misc_files/* $DATA_DIR/	
fi

if [ -d "bash_profile" ]; then
	sudo mkdir -p ${DATA_DIR}/home/root/.profile.d/
	sudo cp -R bash_profile/* ${DATA_DIR}/home/root/.profile.d/
fi

################################################################################
# make an IPK
################################################################################

if $MAKE_IPK; then
	echo "starting building IPK package"

	## make a folder dedicated to IPK building and make the required version file
	mkdir $IPK_DIR
	echo "2.0" > $IPK_DIR/debian-binary

	## add tar archives of data and control for the IPK package
	cd $CONTROL_DIR/
	tar --create --gzip -f ../../$IPK_DIR/control.tar.gz *
	cd ../../
	cd $DATA_DIR/
	tar --create --gzip -f ../../$IPK_DIR/data.tar.gz *
	cd ../../

	## update version with timestamp if enabled
	if $USETIMESTAMP; then
		dts=$(date +"%Y%m%d%H%M")
		VERSION="${VERSION}_${dts}"
		IPK_NAME=${PACKAGE}_${VERSION}.ipk
		echo "new version with timestamp: $VERSION"
	fi

	## use ar to make the final .ipk and place it in the repository root
	ar -r $IPK_NAME $IPK_DIR/control.tar.gz $IPK_DIR/data.tar.gz $IPK_DIR/debian-binary
fi

################################################################################
# make a DEB package
################################################################################

if $MAKE_DEB; then
	echo "starting building Debian Package"

	## make a folder dedicated to IPK building and copy the requires debian-binary file in
	mkdir $DEB_DIR

	## copy the control stuff in
	cp -rf $CONTROL_DIR/ $DEB_DIR/DEBIAN
	cp -rf $DATA_DIR/*   $DEB_DIR

	## update version with timestamp if enabled
	if $USETIMESTAMP; then
		dts=$(date +"%Y%m%d%H%M")
		sed -E -i "s/Version.*/&-$dts/" $DEB_DIR/DEBIAN/control
		VERSION="${VERSION}-${dts}"
		echo "new version with timestamp: $VERSION"
	fi

	DEB_NAME=${PACKAGE}_${VERSION}_arm64.deb
	dpkg-deb --build ${DEB_DIR} ${DEB_NAME}

fi

echo "DONE"
