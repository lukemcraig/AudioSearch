db.getCollection("songs").aggregate(

	// Pipeline
	[
		// Stage 1
		{
			$group: {
			_id: "$artist"
			}
		},

	]

	// Created with Studio 3T, the IDE for MongoDB - https://studio3t.com/

);
