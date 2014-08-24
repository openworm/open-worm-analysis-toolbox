# -*- coding: utf-8 -*-
"""
Two helper functions for use in calculating worm statistics

Functions
---------------------------------------    
fexact
swtest

Notes
---------------------------------------    
Formerly:
SegwormMatlabClasses / +seg_worm / +stats / +helpers / fexact.m  (405 lines)
SegwormMatlabClasses / +seg_worm / +stats / +helpers / swtest.m  (260 lines)

TODO: Look into whether fexact and swtest are already present in scipy

"""

function [pval x K P xx fx pperm] = fexact( varargin )
    """
    %FEXACT Fisher's exact test
    %p = fexact([1 1 1 1 0 0 0]',[1 1 1 0 0 0 0]','tail','r')
    %
    % Fisher's exact test is a statistical test used to compare binary outcomes
    % between two groups. For example, a laboratory test might be positive or negative
    % and we may be interested in knowing whether there is a difference in frequency of positive
    % results among people with a certain disease (cases) compared to those in healthy
    % (controls). The test is applicable whenever there are independent tests conducted on two
    % groups. A crosstable of results (pos/neg) and status (case/control) produces a 2x2 contingency
    % table shown below.
    %       cases  controls
    %  pos   a        c       K
    %  neg   b        d       -
    % total  N        -       M
    %
    % a,b,c,d are the numbers of positive and negative results for the cases
    % and controls. K, N, M are the row, column and grand totals.
    %
    % This function returns a pvalue against the hypothesis that the number of
    % positive cases (and by extension other cells in the table) observed or a more extreme 
    % distribution might have occured by chance given K, N,and M.
    %
    % usage
    %   p = fexact(X,y)
    %       y is a vetor of status (1=case/0=control). X is a MxP matrix of
    %       binary results (pos=1/neg=0). P can be very large for
    %       genotyping assays, each of which can be considered a different way
    %       to categorize the cases and controls.
    %
    %  p = fexact( a,M,K,N, options)
    %      M and N are constants described above. a and K are P-vectors. No checks for valid
    %      input are made.
    %
    %  [p a K C] = fexact(X,y);
    %       also returns vector a. a(i) is the upper left square of the contingency
    %       corresponding to the ith column of X crosstabulated by y. K is a
    %       vector containing a+b for the ith column of X.
    %       C is a lookup table for CDFs that can be used in subsequent calls
    %       to fexact to further improve performance.
    %       C( x(i)+1, K(i)+1) is the cdf for the tail specified in
    %       options. 
    %       NB. this lookup table is only for the given M and N values
    %
    %  p = fexact( a,M,K,N,C]
    %      C is a lookup table that improve performance. Intended to be
    %      used with permutation testing where 1000s of calls are made.
    %      the option "tail" is ignored if C is provided, as it implies
    %      whatever tail was used to generate them. 
    %
    %  p = fexact(..., 'options', values)
    %      options: 
    %      test  'f', 'l'. Sets the test to be either f(exact) or
    %            l(iebermeister). Chris Rorden added this functionality for
    %            single-sided tests. 
    %      tail  l(eft)','r(ight)', 'b(oth)'
    %            left tail tests a negative association. p( x<=a, M, K, N)
    %            right tail tests a positive association. p( x<=b, M, K, M-N)
    %            both (default) is either a positive or negative association.
    %            this is the sum from the left and right tails of the
    %            distribution including all x where p(x|M,K,N) is less than
    %            or equal p(a|M,K,N).
    %      perm  Q, where Q is an integer. Permutation testing when X has multiple Columns. 
    %             to correct for multiple testing. The entire set of tests is repeated Q*repsz
    %            times with permuted y variables. The reported p-values are 
    %            corrected for multiple tests by interpolating into the emprical 
    %            distribution of the minimum p-value obtained from each Q*repsz rounds .
    %            To use this option the X and y calling convention must be used
    %            (as opposed to (a,M,K,N,...) convention.
    %            and Q must be greater than 1.
    %      repsz S, where S is an integer. Used with the PERM option to specify 
    %            the number of replicates to compute at one time (default=100). 
    %            Larger S is faster but requires more memory. 
    %
    %
    % NOTES and LIMITATIONS:
    %      This function is extremely fast when doing multiple tests and is
    %      acceptable with a small number of tests. However, it uses large
    %      tracks of memory. On my 2Gb home Intel Core2 Quad Core Vista machine
    %      this function does 250,000 tests with 100 observations each in 0.10
    %      seconds. I run out of memory when I do more
    %
    % example
    %  x = unidrnd(2, 200, 1000)-1;
    %  y = unidrnd(2,200,1)-1;
    %  p = fexact( x, y );                  % generates p-values for 1000 tests in x
    %  p1 = fexact( x, y, 'perm', 10); % reports p-value relative to empirial
    %                                               % cdf, since these are randomly generated 
    %                                               % the values in p1 should be near  1
    %   
    % References:
    %   Seneta E, Phipps MC (2001) On the Comparison of Two Observed Frequencies. Biometrical Journal, 43(1):23?43, 2001.
    %   Phipps MC (2003)Inequalities between hypergeometric tails. Journal of Applied Mathematics and Decision Sciences. 7(3): 165-174
    %   http://www.emis.de/journals/HOA/ADS/Volume7_3/174.pdf
    %
    % $Id: fexact.m 825 2013-12-05 15:55:16Z mboedigh $
    % Copyright 2012 Mike Boedigheimer
    % Amgen Inc.
    % Department of Computational Biology
    % mboedigh@amgen.com
    """
    
    p = inputParser;
    p.addRequired('A');
    p.addOptional('y',[]);
    p.addOptional('K',[]);
    p.addOptional('N',[]);
    p.addOptional('C',[]);
    p.addOptional('tail', 'b', @(c)ismember( c, {'b','l','r'} ));
    p.addOptional('perm',  0, @(x) isnumeric(x)&isscalar(x));
    p.addOptional('repsz', 100, @(x) isnumeric(x)&isscalar(x));
    p.addOptional('test', 'f', @(c)ismember( c, {'f','l'} )); %CR: Fisher or Liebermeister test
    p.parse(varargin{:});
    P = p.Results.C;
    if isempty(p.Results.N)     % using fexact(X,y,options)
        X = p.Results.A;
        y = p.Results.y;
        if (~islogical(y) && ~all( ismember( y, [0 1])) ) 
            error('linstats:fexact:InvalidArgument', 'y must be in (0,1)' );
        end
        if (~islogical(X) && ~all( ismember( X(:), [0 1])) ) 
            error('linstats:fexact:InvalidArgument', 'X must be in (0,1)' );
        end
    
        y = logical(y);
        N = sum(y);           % number of cases
        M = length(y);
        K = sum(X,1)';
        x = X'*sparse(y);  % in unofficial testing using timeit. This was faster than sum(X(y==1,:))
        % and faster than sum(bsxfun(@eq,X,y))
    else % using fexact( a,M,K,N ...)
        x = p.Results.A; 
        M = p.Results.y;
        N = p.Results.N;
        K = p.Results.K;
    end
    
    
    switch p.Results.tail
        case 'l'; tail = -1;
        case 'r'; tail = 1;
        case 'b'; tail = 2;
    end
    
    if p.Results.test == 'l' %CR added support for Liebermeister Test
        %Seneta E, Phipps MC (2001) On the Comparison of Two Observed Frequencies. Biometrical Journal, 43(1):23?43, 2001.
        %Phipps MC (2003)Inequalities between hypergeometric tails. Journal of Applied Mathematics and Decision Sciences. 7(3): 165-174
        %http://www.emis.de/journals/HOA/ADS/Volume7_3/174.pdf
        %p = fexact(5,31,15,6,'tail','r') %see Phipps p170-> 0.0721
        %p = fexact(5,31,15,6,'tail','r','test','l') %see Phipps p170-> 0.0345
        %p = fexact(10,31,15,25, 'tail','l')
        %p = fexact(10,31,15,25, 'tail','l','test','l')
        if tail == -1 %left tail: increment b,c
            %x = x+1;
            disp('left');
            N = N +1;
            K = K +1;
            M = M + 2;
        elseif tail == 1 %right tail: increment a,d
            x = x+1; 
            N = N +1;
            K = K +1;
            M = M + 2;
        else
            disp('Computing Fisher Exact values: Liebermeister measure only described for computing right tail.');
        end
    end
    
    if N==0 || N==M
        pval = ones(1,length(x));
        return;
    end;
    
    if isscalar(M) && isscalar(N) && isscalar(K) && isscalar(x) && ~(p.Results.perm > 1)
        pval = doTest( x, M, K, N, tail );
        pval(pval>1) = 1;   % fix roundoff errors
        return;
    end
    
    if isempty(P)
        P = getLookup( M,N,tail);
    end
    
    pval = P( sub2ind(size(P), x+1, K+1));
    
    
    if p.Results.perm > 1
        pperm   = doPerm(X,y,M,K,P,p.Results.perm,p.Results.repsz);
        [fx xx] = mecdf(pperm(:));
        xx(end) = max(pval);
        pval    = interp1( xx, fx, pval);
    end
    
    pval(pval>1) = 1;   % fix roundoff errors


"""
##########################################################
"""


function pperm = doPerm(X,y,M,K,P,nperms,repsz)
    """
    % returns pperm a matrix with each element representing the smallest p
    % value from a complete study where the rows of y have been randomly
    % permuted
    """
    pperm = ones(repsz,nperms); % each element is from a permutation test. It is the minimum pvalue among all variables (columns of X)

    """    
    % sub2ind was previously 85% of the execution time. 
    % this method computes the index directly rather than
    % computing x and then calliung sub2ind. It is much
    % faster because it uses the fast multiple engine
    % precompute K*nrows +1, then add x after it is calculated
    """
    X = [ K*size(P,1)+1 X' ]; 
    B = ones(1,repsz);
    
    for permi = 1:nperms
        % generate new random vector y, by shuffling the rows (do not change
        % the counts of M,K or M)
        [i i] = sort( randn(M,repsz) ); % note randn was faster than rand when I checked
        pperm(:,permi) = min( P( X*[B;y(i)] ),[], 1); % find the smallest p-value among all separate tests for each permutation
    end

"""
##########################################################
"""


function P = doTest( x, M, N, K, tail):
    minx = max(0,N+K-M);
    maxx = min(N,K);
    if x < minx
       error( 'x cannot be smaller than max(0,N+K-M)' );
    end
    if x > maxx;
       error( 'x cannot be larger than min(N,K)');
    end
    %log p(x|M,N,K) = Q(K,x)+Q(M-K,N-x)-Q(M,N) where Q(n,k) = lnchoosek(n,k)
    if tail == -1
       a = minx:x;
       logP = lnchoosek( K, a) + lnchoosek(M-K,N-a) - lnchoosek(M,N);
    elseif tail == 1
       a = x:maxx;
       logP = lnchoosek( K, a) + lnchoosek(M-K,N-a) - lnchoosek(M,N);
    else
       a = minx:maxx;
       logP = lnchoosek( K, a) + lnchoosek(M-K,N-a) - lnchoosek(M,N);
       pcrit = logP(x-minx+1);
       k = (logP - pcrit) <= M*eps(max(logP));
       logP  = logP(k);
    end
    P = sum(exp(logP));

"""
##########################################################
"""

function P = getLookup( M, N, tail)
    F = gammaln( 1:M+1);  % used to compute factorials. For small problems
    % It is overkill to generate all factorials up
    % to M, but it takes on .001 seconds to compute
    % all of them to 1..10,001, so we can afford
    % it. Memory-wise may be more of an issue, and
    % a reason to
    
    % lookup value for lnchoosek(a,b) in matrix L for all 1<=b<=a<=M
    % to lookup log nchoosek(n,k) use L(n+1,k+1);
    [k n ind] = tri2sqind(M);
    L = zeros(M);
    L(ind) = lnchoosek( n, k, F);
    L = blkdiag(0,L');      % pad first row and col with 0s for n=0, k=0
    
    
    %log p(x|M,N,K) = Q(K,x)+Q(M-K,N-x)-Q(M,N) where Q(n,k) = lnchoosek(n,k)
    
    % create a lookup table for Log P
    % to lookup log p(x,|M,N,K) use P(x+1,K+1);
    % P initially contains 1s for valid values of x
    % x is in [min( 0, N+K-M), ..., min(N,K)]
    % min(N,K) is on the diagonal. the first
    % column that contains a 0 in the first row will be when the minimum legal
    % value of x is 1, so N+K-M == 1. K will equal M-N+1, which is the column
    % M-N+2
    P = spdiags(ones(N+1,M+1),0:M-N,N+1,M+1); 
    [Xind Kind] = find(P);
    MKind = M-Kind+2;
    NXind = N-Xind+2;
    
    q = sub2ind(size(P),Xind,Kind);
    P = nan(size(P));
    P(q) = exp( L(sub2ind(size(L),Kind,Xind))    + ... Q(K,x)
        L(sub2ind(size(L),MKind,NXind))  + ... Q(M-K, N-x)
        -L(M+1,N+1));                     % ... Q(M,N)
    if tail == -1
        P = cumsum(tril(P, M-N));
    elseif tail == 1
        P = flipud(triu(P));
        P = cumsum(P);
        R = N-(-1:size(P,1)-2)';
        P = P(R,:);
    else
        % This function has been tested against two reference algorithm using all M
        % and N up to 30. The reference algorithms differ in how they find the
        % find the tail on the other side of the distribution.
        [q r] = size(P);
        [P idx] = sort(P,1);     % sort order, called rank, will be transformed to rank below
        ties = [abs(diff(P)) < M*numel(P)*eps(P(1:end-1,:));
            zeros(1,r)];
        R = repmat( (1:q)',1,r);
        R(col2ind(idx)) = R+ties;
        P = cumsum(P,1);          % reuse P to store cumulative P
        P = P(col2ind(R));
    end

"""
##########################################################
"""

function q = lnchoosek( n, k, f)
    % lnchoosek natural log of nchoosek
    %
    % usage
    %       q = lnchoosek( n, k );
    % usage
    %       q = lnchoosek( n, k, f); where f is the gammaln(1:max(n,k)+1);
    q    = n-k;       % allocate space same size as n
    i    = q>0;       % only compute for valid n and k
    
    if nargin < 3
        % q is 0 when n = k because log nchoosek(a,a) = 1 for any a >= 0
        % otherwise it is given by ...
        q(i) = -log(q(i)) - betaln(k(i)+1,q(i));     % otherwise this is it
    else
        q(i) =  f(n(i)+1) - f(k(i)+1) - f( q(i)+1 );
    end
    
    q(~i) = 0;

"""
##########################################################
"""

function i = col2ind(order,siz)
    """
    %col2ind converts column specific 1-based indices to element based indices
    %i = col2ind(A, siz)
    %   return i, a set of integer indices into a m x n matrix
    %   A is a p x n set of column specific indices as you'd get from the second
    %   output of sort
    %   size is optional.
    %        if present, P is set to SIZ(1), otherwise
    %        P is set to the size of the first dimension of ORDER,
    %
    % example
    %   [xsort order] = sort(x);
    %   wrong = x(order);    % this isn't what you want
    %   xsort = x(col2ind(order));   % this is
    """
    [m p] = size(order);
    if nargin < 2
        q = m;
    else
        q = siz(1);
    end
    oset = 0:q:(q*p-1);
    i = order + repmat( oset, m, 1 );

"""
##########################################################
"""

function [i,j,k] = tri2sqind( m, k )
    """
    %TRI2SQIND subscript and linear indices for upper tri portion of matrix
    %
    % get indices into a square matrix for a vector representing a the upper
    % triangular portion of a matrix such as those returned by pdist.
    %
    % [i,j,k] = tri2sqind( m, k )
    %  If V is a hypothetical vector representing the upper triangular portion
    %  of a matrix (not including the diagonal) and
    %  M is the size of a square matrix and
    %  K is an optional vector of indices into V then tri2sqind returns
    %  (i,j) the subscripted indices into the equivalent square matrix.
    %  K is an integer index into the equivalent square matrix
    %
    % Example
    %  X = randn(5, 20);
    %  Y = pdist(X, 'euclidean');
    %  [i,j,k] = tri2sqind( 5 );
    %  S = squareform(Y);
    %  isequal( Y(:), S(k) );
    %  Z = zeros(5);
    %  Z(k) = Y;
    """
    max_k = m*(m-1)/2;
    
    if ( nargin < 2 )
        k = (1:max_k)';
    end;
    
    if any( k > max_k )
        error('linstats:tri2sqind:InvalidArgument', 'ind2subl:Out of range subscript');
    end;
    
    
    i = floor(m+1/2-sqrt(m^2-m+1/4-2.*(k-1)));
    j = k - (i-1).*(m-i/2)+i;
    k = sub2ind( [m m], i, j );

"""
##########################################################
"""

function [fx x] = mecdf( y )
    """
    % MECDF my ECDF function to avoid dependendencies on stats toolbox
    % assumes 0 <= y <= 1
    % x always includes 0 and fx(x>1) = 1;
    """
    n = length(y);
    [x i] = unique(sort(y));
    fx = linspace( 0, 1, n+1 )';
    fx = fx(i+1);
    if x(1)>0
        x = [0;x];
        fx = [0;fx];
    end
    
    if x(end) < 1    % set up the maximum x value as Inf, so interp1 never fails 
        x  = [x;Inf];
        fx = [fx;1];
    end
    
    
    
    
"""
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
"""


function [H, pValue, W] = swtest(x, alpha, tail)
    """
    %SWTEST Shapiro-Wilk parametric hypothesis test of composite normality.
    %   [H, pValue, SWstatistic] = SWTEST(X, ALPHA, TAIL) performs
    %   the Shapiro-Wilk test to determine if the null hypothesis of
    %   composite normality is a reasonable assumption regarding the
    %   population distribution of a random sample X. The desired significance 
    %   level, ALPHA, is an optional scalar input (default = 0.05).
    %   TAIL indicates the type of test (default = 1).
    %
    %   The Shapiro-Wilk hypotheses are: 
    %   Null Hypothesis:        X is normal with unspecified mean and variance.
    %      For TAIL =  0 (2-sided test), alternative: X is not normal.
    %      For TAIL =  1 (1-sided test), alternative: X is upper the normal.
    %      For TAIL = -1 (1-sided test), alternative: X is lower the normal.
    %
    %   This is an omnibus test, and is generally considered relatively
    %   powerful against a variety of alternatives.
    %   Shapiro-Wilk test is better than the Shapiro-Francia test for
    %   Platykurtic sample. Conversely, Shapiro-Francia test is better than the
    %   Shapiro-Wilk test for Leptokurtic samples.
    %
    %   When the series 'X' is Leptokurtic, SWTEST performs the Shapiro-Francia
    %   test, else (series 'X' is Platykurtic) SWTEST performs the
    %   Shapiro-Wilk test.
    % 
    %    [H, pValue, SWstatistic] = SWTEST(X, ALPHA, TAIL)
    %
    % Inputs:
    %   X - a vector of deviates from an unknown distribution. The observation
    %     number must exceed 3 and less than 5000.
    %
    % Optional inputs:
    %   ALPHA - The significance level for the test (default = 0.05).
    %
    %   TAIL  - The type of the test (default = 1).
    %  
    % Outputs:
    %  SWstatistic - The test statistic (non normalized).
    %
    %   pValue - is the p-value, or the probability of observing the given
    %     result by chance given that the null hypothesis is true. Small values
    %     of pValue cast doubt on the validity of the null hypothesis.
    %
    %     H = 0 => Do not reject the null hypothesis at significance level ALPHA.
    %     H = 1 => Reject the null hypothesis at significance level ALPHA.
    %
    
    %
    % References: Royston P. "Algorithm AS R94", Applied Statistics (1995) Vol. 44, No. 4.
    %   AS R94 -- calculates Shapiro-Wilk normality test and P-value
    %   for sample sizes 3 <= n <= 5000. Handles censored or uncensored data.
    %   Corrects AS 181, which was found to be inaccurate for n > 50.
    %
    """
    
    %
    % Ensure the sample data is a VECTOR.
    %
    
    if numel(x) == length(x)
        x  =  x(:);               % Ensure a column vector.
    else
        error(' Input sample ''X'' must be a vector.');
    end
    
    %
    % Remove missing observations indicated by NaN's and check sample size.
    %
    
    x  =  x(~isnan(x));
    
    if length(x) < 3
       error(' Sample vector ''X'' must have at least 3 valid observations.');
    end
    
    if length(x) > 5000
        warning('Shapiro-Wilk test might be inaccurate due to large sample size ( > 5000).');
    end
    
    %
    % Ensure the significance level, ALPHA, is a 
    % scalar, and set default if necessary.
    %
    
    if (nargin >= 2) && ~isempty(alpha)
       if numel(alpha) > 1
          error(' Significance level ''Alpha'' must be a scalar.');
       end
       if (alpha <= 0 || alpha >= 1)
          error(' Significance level ''Alpha'' must be between 0 and 1.'); 
       end
    else
       alpha  =  0.05;
    end
    
    %
    % Ensure the type-of-test indicator, TAIL, is a scalar integer from 
    % the allowable set {-1 , 0 , 1}, and set default if necessary.
    %
    
    if (nargin >= 3) && ~isempty(tail)
       if numel(tail) > 1
          error('Type-of-test indicator ''Tail'' must be a scalar.');
       end
       if (tail ~= -1) && (tail ~= 0) && (tail ~= 1)
          error('Type-of-test indicator ''Tail'' must be -1, 0, or 1.');
       end
    else
       tail  =  1;
    end
    
    % First, calculate the a's for weights as a function of the m's
    % See Royston (1995) for details in the approximation.
    
    x       =   sort(x); % Sort the vector X in ascending order.
    n       =   length(x);
    mtilde  =   norminv(((1:n)' - 3/8) / (n + 0.25));
    weights =   zeros(n,1); % Preallocate the weights.
    
    if kurtosis(x) > 3
        
        % The Shapiro-Francia test is better for leptokurtic samples.
        
        weights =   1/sqrt(mtilde'*mtilde) * mtilde;
    
        %
        % The Shapiro-Francia statistic W is calculated to avoid excessive rounding
        % errors for W close to 1 (a potential problem in very large samples).
        %
    
        W   =   (weights' * x) ^2 / ((x - mean(x))' * (x - mean(x)));
    
        nu      =   log(n);
        u1      =   log(nu) - nu;
        u2      =   log(nu) + 2/nu;
        mu      =   -1.2725 + (1.0521 * u1);
        sigma   =   1.0308 - (0.26758 * u2);
    
        newSFstatistic  =   log(1 - W);
    
        %
        % Compute the normalized Shapiro-Francia statistic and its p-value.
        %
    
        NormalSFstatistic =   (newSFstatistic - mu) / sigma;
        
        % the next p-value is for the tail = 1 test.
        pValue   =   1 - normcdf(NormalSFstatistic, 0, 1);
        
    else
        
        % The Shapiro-Wilk test is better for platykurtic samples.
    
        c    =   1/sqrt(mtilde'*mtilde) * mtilde;
        u    =   1/sqrt(n);
    
        PolyCoef_1   =   [-2.706056 , 4.434685 , -2.071190 , -0.147981 , 0.221157 , c(n)];
        PolyCoef_2   =   [-3.582633 , 5.682633 , -1.752461 , -0.293762 , 0.042981 , c(n-1)];
    
        PolyCoef_3   =   [-0.0006714 , 0.0250540 , -0.39978 , 0.54400];
        PolyCoef_4   =   [-0.0020322 , 0.0627670 , -0.77857 , 1.38220];
        PolyCoef_5   =   [0.00389150 , -0.083751 , -0.31082 , -1.5861];
        PolyCoef_6   =   [0.00303020 , -0.082676 , -0.48030];
    
        PolyCoef_7   =   [0.459 , -2.273];
    
        weights(n)   =   polyval(PolyCoef_1 , u);
        weights(1)   =   -weights(n);
    
        % Special attention when n=3 (this is a special case).
        if n == 3
            weights(1)  =   0.707106781;
            weights(n)  =   -weights(1);
        end
    
        if n >= 6
            weights(n-1) =   polyval(PolyCoef_2 , u);
            weights(2)   =   -weights(n-1);
        
            count  =   3;
            phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2 - 2 * mtilde(n-1)^2) / ...
                    (1 - 2 * weights(n)^2 - 2 * weights(n-1)^2);
        else
            count  =   2;
            phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2) / ...
                    (1 - 2 * weights(n)^2);
        end
    
        %
        % The vector 'WEIGHTS' obtained next corresponds to the same coefficients
        % listed by Shapiro-Wilk in their original test for small samples.
        %
    
        weights(count : n-count+1)  =  mtilde(count : n-count+1) / sqrt(phi);
    
        %
        % The Shapiro-Wilk statistic W is calculated to avoid excessive rounding
        % errors for W close to 1 (a potential problem in very large samples).
        %
    
        W   =   (weights' * x) ^2 / ((x - mean(x))' * (x - mean(x)));
    
        %
        % Calculate the significance level for W (exact for n=3).
        %
    
        newn    =   log(n);
    
        if (n > 3) && (n <= 11)
        
            mu      =   polyval(PolyCoef_3 , n);
            sigma   =   exp(polyval(PolyCoef_4 , n));    
            gam     =   polyval(PolyCoef_7 , n);
        
            newSWstatistic  =   -log(gam-log(1-W));
        
        elseif n >= 12
        
            mu      =   polyval(PolyCoef_5 , newn);
            sigma   =   exp(polyval(PolyCoef_6 , newn));
        
            newSWstatistic  =   log(1 - W);
        
        elseif n == 3
            mu      =   0;
            sigma   =   1;
            newSWstatistic  =   0;
        end
    
        %
        % Compute the normalized Shapiro-Wilk statistic and its p-value.
        %
    
        NormalSWstatistic       =   (newSWstatistic - mu) / sigma;
        
        % The next p-value is for the tail = 1 test.
        pValue       =   1 - normcdf(NormalSWstatistic, 0, 1);
    
        % Special attention when n=3 (this is a special case).
        if n == 3
            pValue  =   1.909859 * (asin(sqrt(W)) - 1.047198);
            NormalSWstatistic =   norminv(pValue, 0, 1);
        end
        
    end
    
    % The p-value just found is for the tail = 1 test.
    if tail == 0
        pValue = 2 * min(pValue, 1-pValue);
    elseif tail == -1
        pValue = 1 - pValue;
    end
    
    %
    % To maintain consistency with existing Statistics Toolbox hypothesis
    % tests, returning 'H = 0' implies that we 'Do not reject the null 
    % hypothesis at the significance level of alpha' and 'H = 1' implies 
    % that we 'Reject the null hypothesis at significance level of alpha.'
    %
    
    H  = (alpha >= pValue);
    
