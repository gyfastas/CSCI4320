#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

// Result from last compute of world.
unsigned char *g_resultData=NULL;

// Current state of world. 
unsigned char *g_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight

static inline void gol_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // calloc init's to all zeros
    g_data = calloc( g_dataLength, sizeof(unsigned char));
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void gol_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    // set all rows of world to true
    for( i = 0; i < g_dataLength; i++)
    {
	g_data[i] = 1;
    }
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    // set first 1 rows of world to true
    for( i = 10*g_worldWidth; i < 11*g_worldWidth; i++)
    {
	if( (i >= ( 10*g_worldWidth + 10)) && (i < (10*g_worldWidth + 20)))
	{
	    g_data[i] = 1;
	}
    }
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    g_data[0] = 1; // upper left
    g_data[worldWidth-1]=1; // upper right
    g_data[(worldHeight * (worldWidth-1))]=1; // lower left
    g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    g_data[0] = 1; // upper left
    g_data[1] = 1; // upper left +1
    g_data[worldWidth-1]=1; // upper right
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
{
    switch(pattern)
    {
    case 0:
	gol_initAllZeros( worldWidth, worldHeight );
	break;
	
    case 1:
	gol_initAllOnes( worldWidth, worldHeight );
	break;
	
    case 2:
	gol_initOnesInMiddle( worldWidth, worldHeight );
	break;
	
    case 3:
	gol_initOnesAtCorners( worldWidth, worldHeight );
	break;

    case 4:
	gol_initSpinnerAtCorner( worldWidth, worldHeight );
	break;

    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
}

static inline void gol_swap( unsigned char **pA, unsigned char **pB)
{
    // Swap the pointers of pA and pB.
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;

}
 
static inline unsigned int gol_countAliveCells(unsigned char* data, 
					   size_t x0, 
					   size_t x1, 
					   size_t x2, 
					   size_t y0, 
					   size_t y1,
					   size_t y2) 
{
  
    // You write this function - it should return the number of alive cell for data[x1+y1]
    // There are 8 neighbors - see the assignment description for more details.
    int aliveCell = 0;
    // up left
    aliveCell += data[x0 + y0];
    // up 
    aliveCell += data[x1 + y0];
    // up right
    aliveCell += data[x2 + y0];
    // left
    aliveCell += data[x0 + y1];
    // buttom left
    aliveCell += data[x0 + y2];
    // buttom
    aliveCell += data[x1 + y2];
    // buttom right
    aliveCell += data[x2 + y2];
    // right
    aliveCell += data[x2 + y1];
    return aliveCell;
}


// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
static inline void gol_printWorld()
{
    int i, j;

    for( i = 0; i < g_worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	}
	printf("\n");
    }

    printf("\n\n");
}

// This function is added to compute the state of current cell given alive neighbors number
static inline unsigned int gol_computeState(unsigned int aliveCell, unsigned char currentState)
{
    switch(currentState)
    {
    case 0:
        // Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction
        if (aliveCell == 3)
            return 1;
        else
            return 0;
        break;
    case 1:
        //Any live cell with fewer than two live neighbors dies, as if caused by under-population.
        if (aliveCell < 2)
            return 0;
        //Any live cell with two or three live neighbors lives on to the next generation.
        if (aliveCell <= 3)
            return 1;
        //Any live cell with more than three live neighbors dies, as if by over-population.
        else
            return 0;
        break;
    
    default:
        printf("Cell state %u is not 0 or 1 \n", currentState);
        exit(-1);
    }
}

/// Serial version of standard byte-per-cell life.
void gol_iterateSerial(size_t iterations) 
{
  size_t i, y, x;
  size_t y0, y1, y2;
  size_t x0, x2;
  unsigned int aliveCells;

  for (i = 0; i < iterations; ++i) 
    {
      for (y = 0; y < g_worldHeight; ++y) 
	{
	    // set y0, y1 and y2 (the world is like a circle)
        y0 = ((y + g_worldHeight - 1) % g_worldHeight) * g_worldWidth;
        y1 = y * g_worldWidth;
        y2 = ((y + 1) % g_worldHeight) * g_worldWidth;
	    
	    for (x = 0; x < g_worldWidth; ++x) 
	    {
		// set x0, x2, call countAliveCells and compute if g_resultsData[y1 + x] is 0 or 1
        x0 = (x + g_worldWidth - 1) % g_worldWidth;
        x2 = (x + 1) % g_worldWidth;
        aliveCells = gol_countAliveCells(g_data, x0, x, x2, y0, y1, y2);
        g_resultData[y1 + x] = gol_computeState(aliveCells, g_data[y1 + x]);
	    }
	}

      // swap resultData and data arrays
      gol_swap(&g_data, &g_resultData);
    }
}

int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int itterations = 0;
    unsigned int outputOn = 0;
    printf("This is the Game of Life running in serial on a CPU.\n");

    if( argc != 5 )
    {
	printf("GOL requires 3 arguments: pattern number, sq size of the world and the number of itterations, e.g. ./gol 0 32 2 \n");
	exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    itterations = atoi(argv[3]);
    outputOn = atoi(argv[4]);
    
    gol_initMaster(pattern, worldSize, worldSize);
    
    gol_iterateSerial( itterations );
    if (outputOn > 0)
    {
    printf("######################### FINAL WORLD IS ###############################\n");
    gol_printWorld();
    }
    free(g_data);
    free(g_resultData);
    
    return true;
}
