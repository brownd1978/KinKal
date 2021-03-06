if [ $TRAVIS_OS_NAME = 'osx' ]; then
  wget https://root.cern/download/root_v6.22.02.macosx64-10.13-clang100.pkg
  installer -pkg root_v6.22.02.macosx64-10.13-clang100.pkg -target CurrentUserHomeDirectory
  source ~/Applications/root_v6.22.02/bin/thisroot.sh 
  python3 -m pip install --user scons
  export PATH=$HOME/Library/Python/3.7/bin:$PATH
else
  wget --no-check-certificate https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb
  sudo dpkg -i cvmfs-release-latest_all.deb

  sudo apt-get update
  sudo apt-get install cvmfs cvmfs-config-default
  rm -f cvmfs-release-latest_all.deb

  sudo /etc/init.d/autofs stop
  sudo mkdir -p /etc/cvmfs/
  #echo "CVMFS_REPOSITORIES=mu2e.opensciencegrid.org,fermilab.opensciencegrid.org" > default.local
  echo "CVMFS_REPOSITORIES=sft.cern.ch" > default.local
  echo "CVMFS_HTTP_PROXY=DIRECT" >> default.local
  sudo mv default.local /etc/cvmfs/default.local

  sudo cvmfs_config setup
  sudo mkdir -p /cvmfs/sft.cern.ch
  sudo mount -t cvmfs sft.cern.ch /cvmfs/sft.cern.ch || echo "mount may have failed, will continue anyway"

  docker pull centos:7

fi

