import gradio as gr
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('all')
nltk.download("stopwords")
nltk.download('punkt')

# Load the model
svc = joblib.load("SVM_model.joblib")

# Load the vectorizer
vectorizer = joblib.load("vectorizer.joblib")

# Load the label encoder
label_encoder = joblib.load("label_encoder.joblib")


# Load NLTK stopwords and create a set of punctuation characters.
stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)


# Function to preprocess text: lowercase, remove stopwords, punctuation.
def preprocess_text(text):
    # Lowercase the text.
    text = text.lower()

    # Tokenize the text.
    words = word_tokenize(text)

    # Remove stopwords, punctuation.
    filtered_words = [
        word for word in words if word not in stop_words and word not in punctuation
    ]

    # Join the filtered and words back into a single string.
    return " ".join(filtered_words)


def predict(text):
    text = preprocess_text(text)
    text_transformed = vectorizer.transform([text])
    pred_classIndex = svc.predict(text_transformed)
    predicted_class = label_encoder.classes_[pred_classIndex][0]
    return predicted_class


# Define the Gradio interface with a custom description_html
description_html = """
<p>This model was trained by Moaz Eldsouky. You can find more about me here:</p>
<p>GitHub: <a href="https://github.com/MoazEldsouky">GitHub Profile</a></p>
<p>LinkedIn: <a href="https://www.linkedin.com/in/moaz-eldesouky-762288251/">LinkedIn Profile</a></p>
<p>Kaggle: <a href="https://www.kaggle.com/moazeldsokyx">Kaggle Profile</a></p>
<p>This model was trained to classify news articles into five categories: Sport, Tech, Entertainment, Politics, and Business.</p>
<p>You can see how this model was trained on the following Kaggle Notebook:</p>
<p><a href="https://www.kaggle.com/code/moazeldsokyx/news-classification-with-svc-clearly-explained">Kaggle Notebook</a></p>
<p>The training data consisted of articles from the BBC News, and the model utilizes an SVM classifier with an impressive accuracy of <strong>98%</strong>.</p>
<p>Feel free to test the model by providing a news article, and it will predict the corresponding category!</p>
"""


sport_news = """Bruno Fernandes' stunning volley gave Manchester United a much-needed victory as they edged past winless Burnley in the Premier League.

United were in desperate search for inspiration following three straight defeats and received it on 45 minutes courtesy of captain Fernandes, who brilliantly volleyed home a first-time finish from Jonny Evans' lofted pass.

The sublime strike was worthy of winning any game and gave United their third win from six league games this season, leaving Vincent Kompany's side bottom of the table with only a point so far.

Burnley played some delightful football in periods and Zeki Amdouni gave the visitors a scare with a free header that was pushed away by Andre Onana at full stretch.

The Burnley frontman had an even better chance when he was played through by Aaron Ramsey, but a low shot cannoned off the foot of the post.

Evans thought he had given United the lead when he headed in a corner, but the effort was ruled out by the video assistant referee (VAR) for Rasmus Hojlund's block on goalkeeper James Trafford.

Burnley went hunting for an equaliser in the second period and Sander Berge narrowly headed over as United held on for victory.

Follow reaction Man Utd's win at Burnley
How did you rate Burnley's performance? Have your say here
What did you make of Manchester United's display? Send us your views here
Fernandes comes to the fore
United had shipped three or more goals in each of their three consecutive defeats against Arsenal, Brighton and Bayern Munich, and they badly required a positive result to get their faltering season back on track.

Marcus Rashford's effort into the side-netting after just 45 seconds may have given indications that this was a side rejuvenated, but it was another largely insipid and lethargic showing from the Red Devils.

Burnley grew into the game after Rashford's early opportunity and will be left wondering about the outcome had Amdouni converted either of the chances that fell his way in the first half.

Under-fire goalkeeper Onana did superbly to keep out a header from the Swiss striker, who also struck the woodwork after a fine team move.

The hosts were playing some eye-catching football but were undone by a moment of magic from Portuguese midfielder Fernandes on the stroke of half-time.

The skipper, who had tested Trafford earlier, peeled away from his marker before letting Evans' pinpoint pass drop into his path and unleashing an unstoppable volley into the bottom corner.

Fernandes could have netted a second even more spectacular strike in injury-time, but his effort on the hook was kept out by Trafford.

Defender Evans, who rejoined the club this summer, was making his first United start since March 2015 and had headed in, but the goal was chalked off by VAR.

The Northern Ireland international was part of the Leicester side that finished in the bottom three last season and Burnley will be looking to avoid the same fate this time.

They have lost all four home games so far this season and it took them nine matches for their first victory in 2021-22 - a campaign which ended in relegation to the Championship.

"""

entertainment_news = """BBC Sound of 2024: Tyla, Last Dinner Party and Kenya Grace tipped for success
The BBC's Sound of 2024 list, which tips music's most exciting new stars, suggests dance and Afrobeats will dominate the next 12 months.

The 10 nominees include chart-topping drum and bass artist Kenya Grace and South Korean house DJ Peggy Gou.

South Africa's Tyla also makes the cut, hot on the heels of her hit single Water; as does Nigeria's Ayra Starr.

Now in its 22nd year, the list has predicted success for Adele, Stormzy, Wet Leg, Fred Again and PinkPantheress.

The winner and top five will be announced in early January.

BBC Radio 1's Sound of 2024 longlist in full This year's longlist suggests African artists will continue to make headway in the UK, following the success of artists like Burna Boy, Asake, Rema, Tems, Wizkid and Tiwa Savage.

Tyla, who was born and raised in Johannesburg, is already making waves with her Grammy-nominated single Water, which is currently in both the UK and US top 10.

Sun-kissed and sensual, the song showcases her unique take on Amapiano, the popular South African genre that blends Afrobeats with deep house and kwaito music.

Ayra Starr began to receive mainstream recognition in 2022 for her song Rush, prompting Rolling Stone to call her "one of Nigeria's most promising new voices".

Singing in English, Pidgin and Yoruba, she has scored three number one singles in her home country and, after collaborations with WizKid and David Guetta, looks set to replicate that success worldwide.
Other artists on this year's longlist include art-rock five-piece The Last Dinner Party, whose debut single Nothing Matters has been streamed more than 13 million times on Spotify.

Building on their early word-of-mouth success, the group have toured this year with Florence + The Machine and Hozier, and were named one of BBC 6 Music's artists of the year.
They're the only band on the Sound of 2024 longlist, which is dominated by female solo acts, including Caity Baser, who describes her sound as "cheeky, British pop".

The Southampton-born singer already has a devoted fanbase, dubbed "Slaysers", and made headlines after capping her ticket prices at an inflation-busting £11.

Baser is joined by CMAT - dubbed "Ireland's answer to Dolly Parton" by the NME thanks to her whip-smart, emotionally-revealing country-pop songs.

There's also a strong showing for British soul and R&B, led by Oxford-born Stevie Wonder fanatic Elmiene

The 22-year-old had a breakout moment in 2021, when his song Golden was chosen to soundtrack Virgil Abloh's final show for Louis Vuitton.

The show took place just two days after Abloh's tragic death, and Elmiene's mournful, nostalgic ballad seemed to be a tribute to the late fashion designer.

He has since released two EPs of equally emotional soul and collaborated with stars from Stormzy and Jamie Woon to Timbaland and Justin Timberlake.
Another artist referencing classic R&B records of the 1960s and 70s is Olivia Dean.

The London-born singer grew up listening to The Supremes and Aretha Franklin, and channelled those influences into her Mercury Prize-nominated debut album, Messy, earlier this year.

A graduate of the Brit School, she previously sang with Rudimental before striking out on her own.

Leicester-born Sekou first signed a record deal at the age of 16 after he was spotted performing in a car park.

Two years later, he has released a debut EP - Out of Mind - that showcases his rich, sonorous bass voice, and earned himself a nomination for the Brits Rising Star Award (alongside Caity Baser and The Last Dinner Party).

Finally, dance music continues its post-pandemic renaissance, with some assistance from Kenya Grace, the South African-born British singer who scored a number one hit with a song she started in her bedroom.

That track was Strangers - a sinuous drum and bass anthem about disposable dating. Her follow-up single, Paris, has just been "tune of the weekend" on Radio 1.

Korean-born, Berlin-based DJ Peggy Gou completes the 2024 longlist, seven years after releasing her debut single, Day Without Yesterday.

A regular crowd-puller at festivals like Glastonbury and Coachella, she crossed into the mainstream this summer with the feelgood anthem (It Goes Like) Nanana. Her latest single, I Believe In Love Again, is a collaboration with rock legend Lenny Kravitz.
The Sound of 2024 was voted for by more than 140 music industry experts, including representatives from Spotify, Apple Music, Glastonbury Festival, the BBC and former nominees including Jorja Smith, PinkPantheress and Tom Grennan.

To qualify, artists must not have had a UK number one or number two album, or more than two top 10 singles by 12 October 2023.

Singers who have appeared on TV talent shows within the last three years are also ineligible.

The winner will be revealed in the new year on BBC Radio 1, with the top five revealed in reverse order between Monday 1 and Friday 5 January.

Last year's prize was won by girl band Flo, with superstar dance producer Fred Again in second place and drum & bass revivalist Nia Archives placing third.
"""

tech_news = """Meta takes down China-based network of thousands of fake accounts

Meta says it recently removed a network of thousands of fake and misleading accounts based in China.

The users posed as Americans and sought to spread polarising content about US politics and US-China relations.

Among the topics the network posted about were abortion, culture war issues and aid to Ukraine.

Meta did not link the profiles to Beijing officials, but it has seen an increase in such networks based in China ahead of the 2024 US elections.

China is now the third-biggest geographical source of such networks, the company said, behind Russia and Iran.

The recent takedowns were outlined in a quarterly threat report released on Thursday by the parent company of Facebook, Instagram and WhatsApp.

The China-based network included more than 4,700 accounts and used profile pictures and names copied from other users around the world.

The accounts shared and liked each other's posts, and some of the content appeared to be taken directly from X, formerly Twitter.

In some cases the accounts copied and pasted verbatim posts from US politicians - both Republicans and Democrats - including former House Speaker Nancy Pelosi, Michigan Governor Gretchen Whitmer, Florida Governor Ron DeSantis, Reps Matt Gaetz and Jim Jordan, and others.

The network displayed no ideological consistency.

In examples released by Meta, an account in the China-based network reposted the words contained in a tweet earlier this year by Democrat Congresswoman Sylvia Garcia. She criticised Texas's abortion laws and wrote: "Let's remember - abortion is healthcare."

But another account in the network copied-and-pasted a tweet from Republican Representative Ronny Jackson, who wrote: "Taxpayer dollars should NEVER fund travel for abortions."

Meta's report stated: "It's unclear whether this approach was designed to amplify partisan tensions, build audiences among these politicians' supporters, or to make fake accounts sharing authentic content appear more genuine."

The company's moderation rules forbid what Meta calls "co-ordinated inauthentic behaviour" - posts by groups of accounts that work together and use false identities to mislead other users.

Often the content shared by such networks is not false and references accurate news stories from major media outlets. But instead of being used for legitimate comment or debate, the posts are meant to manipulate public opinion, push division and make particular viewpoints seem more popular than they really are.

Meta said the large Chinese network was stopped before it took off among real users.

Ben Nimmo, who leads investigations into inauthentic behaviour on the company's platforms, said such networks "still struggle to build audiences, but they're a warning".

"Foreign threat actors are attempting to reach people across the internet ahead of next year's elections, and we need to remain alert."

The company said it also discovered two smaller networks, one based in China and focusing on India and Tibet, and one based in Russia which posted primarily in English about the invasion of Ukraine and promoted Telegram channels.

Russian networks, which prompted the company to focus on inauthentic campaigns following the 2016 election, have increasingly focused on the war in Ukraine and have attempted to undermine international support for Kyiv, the report said.

Meta also noted that the US government stopped sharing information about foreign influence networks with the company in July, after a federal ruling as part of a legal case over the First Amendment that is now under consideration by the Supreme Court.

The case is part of a larger debate about over whether the US government works with tech companies to unduly restrict the free speech of social media users.


"""

business_news = """Bankruptcy 'opportunity' after student loan crisis

Drowning in debt, more and more Americans are taking advantage of a Biden-era change that has made it easier to have student loans forgiven - if they're willing to apply for bankruptcy.

Elizabeth Hadzic, a divorced mum-of-three, has ideas about what she would do if she weren't facing a mountain of student debt: open her own therapy practice, return to her native Canada, work remotely and spend a month with her grandchild.

A change to the US bankruptcy process could make it all possible.

Last year, the US said it would make it easier for people to free themselves of student loans in bankruptcy, a prospect long considered hopeless.

It is a move with potentially vast implications in a country where more than 43 million people carry student debt, generating a total debt load of more than $1.7tn (£1.34tn), and borrowers often face heavy monthly bills decades after they have finished their education.

Bankruptcy, said Ms Hadzic, "opens up opportunity that I couldn't really see before."

This summer she asked the government to erase more than $100,000 in debt from student loans she took on to train as a therapist. She faces potential monthly bills of more than $1,400 - a sum she said she cannot afford alongside the rest of her expenses.

"I thought I'd be able to pay it off... but what I do for work just doesn't make that volume of money," said the 50-year-old, who worked for community health, prison and homeless programmes before switching to a private company in 2019 to try to earn more.

"In my mind, I kept thinking I'm going to pay this money back until I realized it just wasn't going to happen... I would be paying this the rest of my life."

For decades, student loans in the US have faced a higher bar for forgiveness than other debts, like credit-cards, with borrowers forced to prove "undue hardship" if forced to repay - a term that has led to contentious court battles.

The rules were created to prevent borrowers from taking on big loans with no intention of repaying - and limit the potentially huge cost to the federal government, which is the largest provider of student loans in the US.

But critics say it has led to a system that is unduly harsh, generating horror stories of the government fighting bankrupt single mothers and cancer patients for thousands of dollars in monthly debts they are unable to pay.

Unlike the UK and other parts of the world, monthly student loan payments often bear no relation to a borrower's income; nor do they come with an expiration date.

President Joe Biden, whose most ambitious debt forgiveness plan was blocked by the courts this year, backed changes to the bankruptcy system during his 2020 presidential campaign.
Guidance from the Department of Justice announced last year instructed officials to avoid litigation and agree to discharge the loans if a borrower faces higher expenses than income; is unlikely to be able to pay the loan in the future; and has made an effort to pay.

An estimated 250,000 people with student loans file for bankruptcy in the US each year, and under the new guidelines, about 100,000 could be eligible for some student debt relief, according to estimates by Jason Iuliano, a law professor at the University of Utah.

But so far, only a small number - about 630 - have actually petitioned to discharge their student loans as part of their personal bankruptcy.

The Department of Justice would not say how many of those claims had been resolved, but said some relief had been granted in 99% of those that had. Advocates said that number stood at just a few dozen as of July.

With many lawyers still learning about the changes, John Rao, senior attorney at the National Consumer Law Center, said the programme needed more time to prove itself.

But even if the numbers remain small compared to the problem, he said the impact should not be underestimated.

"There are real stories and people behind those numbers," he said. "While it may only be a couple thousand who might use this, for them, it's changed their lives."

Kestrel O'Conally of Washington filed for relief this spring, hoping for a discharge of more than $600,000 in student debt, including nearly $175,000 in interest, racked up in pursuit of a doctorate in psychology.

With little other debt, the 41-year-old said she had never considered bankruptcy before a friend alerted her to a news article describing the changes.
In her situation, as a renter with few assets, the consequences from bankruptcy, like having credit cards cancelled, seemed like small prices to pay, she said.

"It was a no brainer," she said. "I get my life back."

Those who have pursued relief via bankruptcy typically face months of wait. Aaron Ament, president of the National Student Legal Defense Network, warned those delays could worsen if the idea gains traction.

Petitions in the first 10 months since the change were already about 30% more than the average pre-pandemic year.

Those numbers are expected to accelerate, as borrowers have to resume student loan payments after a three-year pandemic-related pause ended in October.

"It's getting real now in terms of coming up with the money to pay the loans out of their existing budget," said Latife Neu, a student loan and bankruptcy lawyer in Washington state, who said she had seen a steady uptick in inquiries.

Ms Hadzic said her bankruptcy was not technically triggered by her student loans, but by thousands of dollars in credit card debt racked up when her youngest son landed in hospital and she had to take off work.

But her lawyer, Timothy Chambers, said without addressing the student loans as well, he feared she could land in a similar financial crisis.

"I'm hoping this will be able to give people like Elizabeth that absolute fresh start which bankruptcy is supposed to give," he said.

"""

politics_news = """'Labour won't turn on spending taps' says Starmer
Sir Keir Starmer is to warn that the UK will face "huge constraints" on public spending if his party win the next General Election.

On Monday, he will say anyone expecting an incoming Labour government "to quickly turn on the spending taps is going to be disappointed".

Growing the economy is set to be a battleground for both Labour and the Conservatives at the next election.

The Conservatives claim Labour's borrowing plan would raise taxes.

Rishi Sunak, the prime minister, has made economic growth one of his key pledges.

The cost of living in the UK has put the economy at the centre of political debate as inflation and high interest rates put pressure on household budgets.

The UK has not slipped into recession but there have been concerns over weak growth.

The Bank of England's governor, Andrew Bailey, said last week that productivity rates in the UK "concern me a lot".

In a speech to economists and think-tanks later on Monday, Sir Keir will say economic growth "will have to become Labour's obsession if we are to turn around the economy".

But he will argue that decisions taken from the government and previous Conservative administrations for the past 13 years "will constrain what a future Labour government can do".

"The comparison between 2010 and today is instructive," the Labour leader will say.

"Now, debt and interest rates are much higher. Britain's standing is diminished. Growth is stagnant and public services are on their knees."

"Taxes are higher than at any time since the war, none of which was true in 2010. Never before has a British government asked its people to pay so much, for so little."

His comments will come after his party denied claims it could further water down its flagship green prosperity plan.

A senior source had suggested to the BBC that the level of investment previously promised - of £28bn a year - might never be reached.

In June Rachel Reeves, the shadow chancellor, watered the pledge down. Senior source in the Labour leader's office said the decision was made because of the state of the public finances.

However, a Conservative spokesman said Labour's policy "presents a major risk" to the British economy at a time when the cost of borrowing is "so high".

Richard Holden, the Conservative party chairman, said: "The largest 'constraint' to growing the economy would be Labour's £28bn a year borrowing plan - which independent economists warn would see inflation, interest rates and people's taxes rise".

Labour denies abandoning £28bn green pledge
As well as focusing on growth, Sir Keir will outline his plans which include changing "restrictive planning laws and get Britain building again" and creating a "proper industrial strategy drawn up with business".

He will also declare Labour would secure "a new deal to make work pay with increased mental health support, fully-funded plan to cut NHS waiting lists, an end to zero hour contracts, no more fire and rehire, and a real living wage".

The government has announce it will increase the minimum wage by more than a pound to £11.44 next April, but the government's forecaster, the Office for Budget Responsibility (OBR), has said living standards are also not expected to return to pre-pandemic levels until 2027-28.

Sir Keir will say that the current parliament is "on track to be the first in modern history where living standards in this country have actually contracted".

Most recent official figures show the economy failed to grow between July and September, after a succession of interest rate rises that have increased borrowing costs.

Chancellor Jeremy Hunt announced several tax cuts in his autumn statement.

One policy move included making permanent a tax break which allows businesses to deduct the full cost of investing in machinery and equipment from their tax bill.

But the decisions will not prevent taxes staying at their highest level on record and economic growth is forecast to be sluggish.

Sir Keir will also say in his speech that "it's not the case that 'any growth' will do".

"No, we can't be agnostic about the sort of growth we pursue, anymore. The growth we need must better serve working people. And must raise living standards in every community," he will argue.


"""


iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        placeholder="Paste your news article here...", label="Enter BBC news article:"
    ),
    outputs=gr.Textbox(placeholder="Predicted Category", show_label=True),
    description=description_html,  # Include the custom description_html
    allow_flagging="never",
    title="📰🔍 News Categorizer 98 🎯✨",
    examples=[
        [sport_news],
        [tech_news],
        [entertainment_news],
        [politics_news],
        [business_news],
    ],
)


# Launch the Gradio app
iface.launch()
