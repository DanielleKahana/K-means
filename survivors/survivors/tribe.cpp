#include "tribe.h"

void Tribe::init(int maxSurvivors)
{
	_maxSurvivors = maxSurvivors;
	_currentNumOfSurvivors = 0;
	_allSurvivors = new Survivor*[_maxSurvivors];
}

bool Tribe::addSurvivor(Survivor& newSurvivor)
{
	if (_currentNumOfSurvivors < _maxSurvivors)
	{
		_allSurvivors[_currentNumOfSurvivors++] = &newSurvivor;
		return true;
	}
	else
		return false;
}

Survivor** Tribe::getAllSurvivors()
{
	return _allSurvivors;
}

int Tribe::getCurrentNumOfSurvivors() const {
	return _currentNumOfSurvivors;
}

int Tribe::getMaxSurvivors() const {
	return _maxSurvivors;
}

void Tribe::print() const {
	for (int i = 0; i < _currentNumOfSurvivors; i++)
		_allSurvivors[i]->print();
}

void Tribe::free() {
	/*for (int i = 0; i < _currentNumOfSurvivors; i++)
	delete _allSurvivors[i];*/

	delete[]_allSurvivors;
}
