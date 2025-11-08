#include <iostream>
#include <string>
using namespace std;

struct Action{
    string description;
};
struct Node{
    Action data;
    Node* next;
    
    Node(const Action &action){
        data = action;
        next = nullptr;
    }
};
class Stack{
private:
    Node* top;
public:
    Stack(){
        top = nullptr;
    }
    ~Stack(){
        clear();
    }
    void push(const Action &action){
        Node* newNode = new Node(action);
        newNode->next=top;
        top=newNode;
    }
    Action pop(){
        if (isEmpty()){
            return {"No Action"};
        }
        Node* temp = top;
        Action popped = temp->data;
        top = top->next;
        delete temp;
        return popped;
    }
    Action peek() const{
        if(isEmpty()){
            return {"No Action"};
        }
        return top->data;
    }
    
    bool isEmpty() const{
        return top == nullptr;
    }
    void clear(){
        while(!isEmpty()){
            pop();
        }
    }
};

