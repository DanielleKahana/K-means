#include "survivor.h"

void Survivor::setName(const char *name)
{
	if (strlen(name) >= MAX_NAME_SIZE) {
		cout << "The name must contain " << (MAX_NAME_SIZE - 1) << " letters or less." << endl;
		return;
	}

	strcpy(_name, name);
}

void Survivor::setAge(int age)
{
	_age = age;
}

void Survivor::setStatus(Survivor::eFamilyStatus status) {
	_status = status;
}

const char * Survivor::getName() const
{
	return _name;
}

int Survivor::getAge() const
{
	return _age;
}

Survivor::eFamilyStatus Survivor::getStatus() const
{
	return _status;
}

void Survivor::init(const char* name, int age, eFamilyStatus status)
{
	setName(name);
	setAge(age);
	setStatus(status);
}

void Survivor::print() const
{
	cout << "name: " << _name << "\nage: " << _age << "\nstatus: " << _status << "\n";
}
