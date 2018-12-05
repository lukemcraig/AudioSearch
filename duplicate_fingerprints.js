// Stages that have been excluded from the aggregation pipeline query
__3tsoftwarelabs_disabled_aggregation_stages = [

	{
		// Stage 4 - excluded
		stage: 4,  source: {
			$sort: {
			n_offsets:-1
			}
		}
	},
]

db.getCollection("fingerprints").aggregate(

	// Pipeline
	[
		// Stage 1
		{
			$group: {
			_id :{hash:"$hash",songID:"$songID"},
			count: { $sum: 1 },
			offsets:{$addToSet:"offset"},
			}
		},

		// Stage 2
		{
			$addFields: {
			    n_offsets:{$size:"$offsets"}
			}
		},

		// Stage 3
		{
			$sort: {
			count:-1
			}
		},

	]

);
