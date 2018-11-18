#pragma once
#ifndef __TRIBE_H
#define __TRIBE_H

#include "survivor.h"

class Tribe
{
private:
	Survivor **_allSurvivors;
	int _maxSurvivors;
	int _currentNumOfSurvivors;

public:
	void init(int maxSurvivors);

	bool addSurvivor(Survivor& newSurvivor);
	//void setAllSurvivors(Survivor **allSurvivors);
	//void setMaxSurvivors(int maxSurvivors);
	//void setCurrentNumOfSurvivors(int currentNumOfSurvivors);

	/*inline*/ Survivor** getAllSurvivors();
	/*inline*/ int getMaxSurvivors() const;
	/*inline*/ int getCurrentNumOfSurvivors() const;

	void print() const;
	void free();
};

#endif // __TRIBE_H
