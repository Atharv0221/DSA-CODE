First javac file then use java

for more than one java file use javac *.java

sudo apt install openjdk-8-jdk -y

sudo apt update
sudo apt install default-jdk -y      # Ubuntu/Debian
java -version
javac -version


Program 2: CORBA (needs JDK 8)

CORBA was removed from Java in JDK 11, so install JDK 8 and use it for this program only.

bash
sudo apt install openjdk-8-jdk -y
sudo update-alternatives --config java      # pick Java 8
sudo update-alternatives --config javac     # pick Java 8
java -version                               # should show 1.8

Then, from the folder with Hello.idl, HelloImpl.java, Server.java and Client.java:

bash
idlj -fall Hello.idl
javac *.java HelloApp/*.java

Terminal 1:

bash
orbd -ORBInitialPort 1050

Terminal 2:

bash
java Server -ORBInitialPort 1050 -ORBInitialHost localhost

Terminal 3:

bash
java Client -ORBInitialPort 1050 -ORBInitialHost localhost
