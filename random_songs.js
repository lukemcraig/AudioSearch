db.getCollection("songs").aggregate(

	// Pipeline
	[
		// Stage 1
		{
			$sample: {
			    size: 10
			}
		},

	]

	// Created with Studio 3T, the IDE for MongoDB - https://studio3t.com/

);
