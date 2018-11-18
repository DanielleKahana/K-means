#include "survivor.h"
#include "tribe.h"

void readTribe(Tribe& t);
void readSurvivor(Survivor& s);

void main() {
	Tribe t1, t2;

	t1.init(20);
	t2.init(20);

	cout << "Enter data for first tribe:\n";
	readTribe(t1);

	cout << "Enter data for second tribe:\n";
	readTribe(t2);

	t1.print();
	t2.print();

	int currentSurvivors = t1.getCurrentNumOfSurvivors();
	for (int i = 0; i < currentSurvivors; i++)
		delete t1.getAllSurvivors()[i];

	currentSurvivors = t2.getCurrentNumOfSurvivors();
	for (int i = 0; i < currentSurvivors; i++)
		delete t2.getAllSurvivors()[i];

	t1.free();
	t2.free();

	system("pause");
}

void readSurvivor(Survivor& s)
{
	char name[Survivor::MAX_NAME_SIZE];
	cin.ignore(); // clean the buffer
	cout << "Enter your name: ";
	cin.getline(name, Survivor::MAX_NAME_SIZE);

	int age;
	cout << "Enter you age:  ";
	cin >> age;

	int statusNumber;
	cout << "Enter your status: \n 1.Single \n 2.Married \n 3.In A Relationship \n";
	cin >> statusNumber;

	s.init(name, age, (Survivor::eFamilyStatus)statusNumber);
}

void readTribe(Tribe& t) {
	bool toContinue = true;
	int indicator;

	int maxSurvivors = t.getMaxSurvivors();
	for (int i = 0; (i < maxSurvivors) && toContinue; i++)
	{
		Survivor* s = new Survivor;
		readSurvivor(*s);
		t.addSurvivor(*s);

		cout << "Continue? Press 1 for yes 2 for no \n";
		cin >> indicator;

		if (indicator == 2)
			toContinue = false;

	}
}
