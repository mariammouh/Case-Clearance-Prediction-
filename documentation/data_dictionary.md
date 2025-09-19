# LAPD Crime Data Dictionary

## Core Identifiers
- **DR Number**: Division of Records number - unique case identifier
- **Date Reported**: Date the crime was reported to police
- **Date Occurred**: Date the crime actually occurred

## Crime Classification  
- **Part 1-2**: Crime type classification (1 = Major, 2 = Minor)
- **Crime Code**: Numerical code representing specific crime type
- **Crime Description**: Text description of the crime type
- **MO Codes**: Modus Operandi codes describing how crime was committed

## Location Data
- **Area**: Police precinct code
- **Area Name**: Police precinct name  
- **LAT/LON**: Geographic coordinates of crime location
- **Premise Code**: Code for type of location where crime occurred
- **Premise Description**: Description of location type

## Victim Information
- **Victim Age**: Age of the victim
- **Victim Sex**: Gender of the victim (F/M/X)
- **Victim Descent**: Racial descent of the victim

## Case Status
- **Status Description**: Final status of the case (target variable)
- **Weapon Used**: Code indicating if weapon was involved

## Engineered Features
- **time_to_report**: Hours between occurrence and reporting
- **crime_count**: Number of associated crime codes
- **location_cluster**: Geographic cluster assignment
- **mo_embedding_***: Learned embeddings from MO codes