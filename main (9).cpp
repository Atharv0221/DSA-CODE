#include <iostream>
#include "stack.h"
#include <string>
using namespace std;

void showMenu() {
	cout<<"\n--Text Editor Menu--\n";
	cout<<"1. Type text\n";
	cout<<"2. Change Font\n";
	cout<<"3. Change Case\n";
	cout<<"4. New Line\n";
	cout<<"5. Undo\n";
	cout<<"6. Redo\n";
	cout<<"7.Exit\n";
	cout<<"Enter your choice: ";
}

int main() {
	Stack undoStack;
	Stack redoStack;

	int choice;
	string input;
	Action action;

	while(true) {
		showMenu();
		cin>>choice;
		cin.ignore();

		switch(choice) {
		case 1:
			cout<<"Enter text: ";
			getline(cin,input);
			action.description="Typed Text: "+input;
			undoStack.push(action);
			redoStack.clear();
			break;

		case 2:
			cout<<"Enter Font Name: ";
			getline(cin,input);
			action.description="Changed Font to: "+input;
			undoStack.push(action);
			redoStack.clear();
			break;

		case 3:
			cout<<"Enter case (Upppercase/Lowercase): ";
			getline(cin,input);
			action.description="Changed Case to: "+input;
			undoStack.push(action);
			redoStack.clear();
			break;

		case 4:
			action.description="Inserted new line: ";
			undoStack.push(action);
			redoStack.clear();
			break;

		case 5:
			if(!undoStack.isEmpty()) {
				Action undone = undoStack.pop();
				redoStack.push(undone);
				cout<<"Undo performed: "<<undone.description<<endl;
			}
			else {
				cout<<"Nothing to undo.\n";
			}
			break;

		case 6:
			if(!redoStack.isEmpty()) {
				Action redone=redoStack.pop();
				undoStack.push(redone);
				cout<<"Redo performed: "<<redone.description<<endl;
			}
			else {
				cout<<"Nothing to redo\n";
			}
			break;

		case 7:
			cout<<"Exiting text editior\n";
			return 0;

		default:
			cout<<"Invalid Options, Choose again.\n";
		}
	}
	return 0;
}
