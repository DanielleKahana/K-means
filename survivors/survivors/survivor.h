#pragma once
#ifndef __SURVIVORS_H
#define __SURVIVORS_H

#include <iostream>
#include <string.h>
using namespace std;

#pragma warning(disable: 4996)

class Survivor {
public:
	enum eFamilyStatus { SINGLE, MARRIED, IN_A_RELATIONSHIP };
	static const int MAX_NAME_SIZE = 20;

private:
	char _name[MAX_NAME_SIZE];
	int _age;
	eFamilyStatus _status;

public:
	inline void setName(const char *name);
	inline void setAge(int age);
	inline void setStatus(eFamilyStatus status);

	inline const char * getName() const;
	inline int getAge() const;
	inline eFamilyStatus getStatus() const;

	void init(const char* name, int age, eFamilyStatus status);
	void read();
	void print() const;
};

#endif // __SURVIVORS_H
