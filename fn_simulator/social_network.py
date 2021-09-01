import datetime
from enum import IntEnum
import numpy as np
import random
import networkit as nk
import matplotlib.pyplot as plt
import pickle
import copy
import sys
from scipy.spatial import KDTree
import multiprocessing as mp
sys.setrecursionlimit(100000)


ENGAGEMENT_FACTOR=0.1
POLITICAL_ENGAGEMENT_FACTOR=0.1
RETWEET_ENGAGEMENT_THRESHOLD=0
HEALTH_PRECAUTIONS_ALPHA=0.3
FOLLOW_ENGAGEMENT_THRESHOLD=0.8
MEAN_CONTACT_GROUPED=1
CONTACT_MINIMUM_THRESHOLD=0.001
VAXX_PRECAUTION_ALPHA=0.1


def distance(x,y):
    if np.array(x).size!=np.array(y).size:
        raise Exception("Distance Objects do not match size")
    return np.linalg.norm(np.array(x)-np.array(y)) / np.sqrt(len(np.array(x)))

def normal(mean,variance,lower_clip,upper_clip):
    return np.clip(np.random.normal(mean,variance),lower_clip,upper_clip)

def bimodal(mean1,mean2,variance1,variance2,lower_clip,upper_clip):
    choice=random.choice([0,1])
    return  np.clip(choice*normal(mean1,variance1,lower_clip,upper_clip)+(1-choice)*normal(mean2,variance2,lower_clip,upper_clip),lower_clip,upper_clip)

class AgentType(IntEnum):
    Regular=0
    Influencer=1
    Bot=2
    HoneyBot=3
    Authority=4

class FakeNewsState(IntEnum):
    Susceptible=0
    Infected=1
    Recovered=2
    Noninfectable=3
    Unfaked=4

class DiseaseState(IntEnum):
    Susceptible=0
    Infected=1
    Recovered=2
    NonApplicable=3

class Agent:

    def __init__(self,
                 id : int ,
                 type:AgentType,
                 pol_distribution = normal(0,0.3,-1,1),
                 bot_pol_distribution=1,
                 separate_influencer_pol=True):

        self.id=id
        self.type=type
        self.tweet_rate=normal(0.5,0.2,0,1)
        self.est_trust=normal(0.7,0.2,0,1)
        self.vul=normal(0.5,0.2,0,1)
        if separate_influencer_pol and type == AgentType.Influencer:
            self.political=pol_distribution
        else:
            self.political=normal(0,0.3,-1,1)
        self.int_1=normal(0.5,0.2,0,1)
        self.int_2=normal(0.5,0.2,0,1)
        self.int_3=normal(0.5,0.2,0,1)
        self.geographical=[random.uniform(0,1),random.uniform(0,1)]
        self.access_time=random.randint(2,10)
        self.tweets=[]
        self.last_access=-1*self.access_time
        self.news_state=FakeNewsState.Susceptible
        self.health_precautions=normal(0.5,0.2,0,1)
        self.centralized_location_index=-1
        self.disease_state=DiseaseState.Susceptible
        self.changed_state=False
        self.infection_time=-1
        self.recovery_time=normal(5,1,1,10)
        self.transmitted_agents=[]

        if (type==AgentType.HoneyBot) or (type==AgentType.Authority):
            self.access_time=1
            self.news_state=FakeNewsState.Noninfectable


        if  (type==AgentType.Bot):
            self.access_time=1
            self.political=bot_pol_distribution
            self.news_state=FakeNewsState.Noninfectable
            self.disease_state=DiseaseState.NonApplicable

        self.following=[]
        self.followers=[]
        self.following_id=[]


        self.faked_tweets=[]
        self.prev_tweets=[]

        self.vaccine_rate=normal(0.3,0.1,0,1)
        self.contact=0

    def asdict(self):
        edge_trusts=[edge.trust for edge in self.following]

        return {"id": self.id, "type":self.type, "tweet_rate":self.tweet_rate,"est_trust":self.est_trust,"vul":self.vul,"political":self.political,
                "access_time":self.access_time,"news_state":self.news_state,"edge_trust":edge_trusts,"disease_state":self.disease_state}


    def add_tweets(self,twt):

        if type(twt)==list:
            self.tweets+=twt
        else:
            if twt.id not in self.prev_tweets:
                self.tweets.append(twt)
                self.prev_tweets.append(twt.id)


    def clear_tweets(self):
        self.tweets=[]

    def change_news_state(self,news_state: FakeNewsState):
        self.changed_state=True
        self.news_state=news_state
        if news_state==FakeNewsState.Infected:
            self.health_precautions-=HEALTH_PRECAUTIONS_ALPHA
            self.vaccine_rate-=VAXX_PRECAUTION_ALPHA*3

        if news_state==FakeNewsState.Recovered:
            self.health_precautions+=HEALTH_PRECAUTIONS_ALPHA

    def update_political(self,engagement):
        engagement=max(engagement,-1*engagement)
        if self.political>0:
            self.political+=POLITICAL_ENGAGEMENT_FACTOR*engagement
        else:
            self.political-=POLITICAL_ENGAGEMENT_FACTOR*engagement

        self.political=max(-1,min(1,self.political))

    def unfollow(self,edge):
        self.following.remove(edge)
        self.following_id.remove(edge.agentto.id)
        edge.agentto.followers.remove(edge)

    def follow(self,node2):
        if not node2.id in self.following_id:
            new_edge=Edge(self,node2)
            self.following.append(new_edge)
            self.following_id.append(node2.id)
            node2.followers.append(new_edge)
            return new_edge

class Tweet:
    id=0
    @classmethod
    def make_id(self):
        self.id+=1
        return self.id

    def __init__(self,misinformation: int, agent: Agent, time: int, correction: int=0, correction_tweet = None):

        self.id=self.make_id()
        self.misinformation = misinformation
        self.tweeter=agent
        self.correction=correction
        self.correction_tweet=correction_tweet
        self.time=time
        self.political=0
        self.int_1=0
        self.int_2=0
        self.int_3=0

        if self.misinformation:
            self.political=agent.political

        elif self.correction:
            self.political=-1*correction_tweet.political

        else:
            s=random.randint(0,3)
            if s==0:
                self.political = np.random.sample() * agent.political
            else:
                new_val=getattr(agent,f"int_{s}")*np.random.sample()
                setattr(self,f"int_{s}",new_val)


    def calculate_prob(self, stage,node_access_time):
        exp_factor=-1*np.floor((stage-self.time)/node_access_time)

        if self.misinformation:
            return 2*np.exp(exp_factor)
        else:
            return np.exp(exp_factor)


class Edge:
    def __init__(self,
                 agent_1: Agent,
                 agent_2: Agent,
                 bot_edge=0):
        self.agentfrom=agent_1
        self.agentto=agent_2

        self.tweets=[]
        self.bot_edge=bot_edge


        if not bot_edge:
            g_dist=1-distance(agent_1.geographical,agent_2.geographical)
            self.trust=max(g_dist,
                           agent_1.int_1*agent_2.int_1,
                           agent_1.int_2 * agent_2.int_2,
                           agent_1.int_3*agent_2.int_3,
                           agent_1.political*agent_2.political)
        else:
            self.trust=0.5

    def update_edge(self, engagement:float):
        self.trust+=ENGAGEMENT_FACTOR*engagement

        if self.trust>1:
            self.trust=1

class Network:
    def __init__(self, N_common: int,
                      N_influencers: int,
                      N_bots: int,
                      N_honeybots: int,
                      N_authority: int,
                      N_centralized_locale: int,
                      N_disease_infected: int,
                      agent_pol=normal(0,0.3,-1,1),
                      bot_pol=1,
                      influencer_only=False
    ):

        self.finalized=False
        self.Nodes={}
        self.common_nodes=[]
        self.influencer_nodes=[]
        self.bot_nodes=[]
        self.honeybot_nodes=[]
        self.authority_nodes=[]
        self.centralized_locations={}

        total_N=N_common+N_influencers+N_bots+N_honeybots+N_authority
        self.graph=nk.Graph(total_N,directed=True)
        self.idx=0

        for i in range(N_common):
            self.Nodes[self.idx]=Agent(self.idx,AgentType.Regular,agent_pol,bot_pol,influencer_only)
            self.common_nodes.append(self.idx)
            self.idx+=1

        for i in range(N_influencers):
            self.Nodes[self.idx]=Agent(self.idx,AgentType.Influencer,agent_pol,bot_pol,influencer_only)
            self.influencer_nodes.append(self.idx)
            self.idx+=1


        for i in range(N_bots):
            self.Nodes[self.idx]=Agent(self.idx,AgentType.Bot,agent_pol,bot_pol,influencer_only)
            self.bot_nodes.append(self.idx)
            self.idx+=1

        for i in range(N_honeybots):
            self.Nodes[self.idx]=Agent(self.idx,AgentType.HoneyBot,agent_pol,bot_pol,influencer_only)
            self.honeybot_nodes.append(self.idx)
            self.idx+=1

        for i in range(N_authority):
            self.Nodes[self.idx]=Agent(self.idx,AgentType.Authority,agent_pol,bot_pol,influencer_only)
            self.authority_nodes.append(self.idx)
            self.idx+=1


        location_nodes=self.common_nodes+self.influencer_nodes
        np.random.shuffle(location_nodes)

        self.random_draw_nodes=random.sample(self.common_nodes,50)

        for c,nodes_list in enumerate(np.array_split(location_nodes,N_centralized_locale)):
            for agent_idx in nodes_list:
                self.Nodes[agent_idx].centralized_location_index=c

            self.centralized_locations[c]=[self.Nodes[i] for i in nodes_list]

        infected_nodes=random.sample(location_nodes,N_disease_infected)

        for node_idx in infected_nodes:
            self.Nodes[node_idx].disease_state=DiseaseState.Infected

    def add_edges(self,Geographical_proximity: float,
                  Interest_proximity:float,
                  Bot_coverage: float,
                  Authority_coverage:float,
                  save=True,name=None):

        for node in self.Nodes.values():
            int_prox=Interest_proximity
            geo_prox=Geographical_proximity

            if node.type==AgentType.Influencer:

                geo_prox=Geographical_proximity/4
                int_prox=Interest_proximity/4

            if not (node.type==AgentType.Bot or node.type==AgentType.HoneyBot or node.type==AgentType.Authority) :
                for node_2 in self.Nodes.values():
                    if not (node_2.type==AgentType.Bot or node_2.type==AgentType.HoneyBot or node_2.type==AgentType.Authority):

                        if node_2.type==AgentType.Influencer:
                            geo_prox= Geographical_proximity*4
                            int_prox=Interest_proximity*4


                        if (random.randint(0,100)<50) and ( (distance(node_2.geographical,node.geographical)<geo_prox) or (distance([node.int_1,node.int_2,node.int_3],[node_2.int_1,node_2.int_2,node_2.int_3])<int_prox) ):
                            new_edge=Edge(node,node_2)
                            node.following.append(new_edge)
                            node.following_id.append(node_2.id)
                            node_2.followers.append(new_edge)
                            self.graph.addEdge(node.id,node_2.id)

            else:
                if node.type==AgentType.Authority:
                    N_bot_followers=int(Authority_coverage*len(self.Nodes))
                else:
                    N_bot_followers=int(Bot_coverage*len(self.Nodes))

                follower_nodes=random.sample(self.common_nodes+self.influencer_nodes,N_bot_followers)
                for nod_ix in follower_nodes:
                    new_edge=Edge(self.Nodes[nod_ix],node,1)
                    node.followers.append(new_edge)
                    self.Nodes[nod_ix].following.append(new_edge)
                    self.Nodes[nod_ix].following_id.append(node.id)
                    self.graph.addEdge(nod_ix,node.id)

        self.finalized=True

        if save:
            if not name:
                name=datetime.datetime.now().strftime("%m-%d %H-%M")
            self.save_pickle(f"{name}.obj")

    def draw_network(self,fig_name):
        posititons={}
        colors=[]
        for idx,node in self.Nodes.items():
            posititons[idx]=node.geographical
            if node.type==AgentType.Influencer:
                colors.append("red")
            if node.type==AgentType.Bot:
                colors.append("blue")
            if node.type==AgentType.HoneyBot:
                colors.append("green")
            if node.type==AgentType.Regular:
                colors.append("pink")
        nk.viztasks.drawGraph(self.graph,with_labels=True,pos=posititons,node_color=colors,node_size=150,edge_color="grey")
        plt.savefig(fig_name)
        return plt


    def draw_political_network(self,fig_name,stage_no):
        selected_nodes={}
        node_to_selected_node_map={}
        for i, idx in enumerate(self.random_draw_nodes):
            selected_nodes[i]=self.Nodes[idx]
            node_to_selected_node_map[idx]=i

        new_network=nk.Graph(len(selected_nodes),directed=True)

        for node_id,node in selected_nodes.items():
            for idx in node.following_id:
                if node_to_selected_node_map.get(idx):
                    new_network.addEdge(node_id,node_to_selected_node_map[idx])

        positions={}
        colors=[]
        edge_colors=[]

        for node1,node2 in new_network.iterEdges():
            if selected_nodes[node1].political<0:
                edge_colors.append("blue")
            else:
                edge_colors.append("red")


        for idx,node in selected_nodes.items():


            if node.political>0 and stage_no>20:
                colors.append("red")
                if stage_no>20:
                    positions[idx]=[node.political-normal(0.2,0.1,0,1),node.geographical[1]]
                else:
                    positions[idx]=[node.political,node.geographical[1]]
            else:
                colors.append("blue")
                if stage_no>20:
                    positions[idx]=[node.political+normal(0.2,0.1,0,1),node.geographical[1]]
                else:
                    positions[idx]=[node.political,node.geographical[1]]

        fig,ax=plt.subplots(1,figsize=(10,10))
        nk.viztasks.drawGraph(new_network,with_labels=False,pos=positions,node_color=colors,node_size=75,edge_color=edge_colors,ax=ax)
        ax.set_xlim(-1.1,1.1)
        fig.savefig(fig_name)

    def degree_distribution(self,dist_type="all",outdegree=True):
        scores=np.array(nk.centrality.DegreeCentrality(self.graph,outDeg=outdegree).run().scores())
        score_dict={}
        score_dict[AgentType.Regular]=scores[self.common_nodes]
        score_dict[AgentType.Influencer]=scores[self.influencer_nodes]
        score_dict[AgentType.Bot]=scores[self.bot_nodes]
        score_dict[AgentType.HoneyBot]=scores[self.honeybot_nodes]
        score_dict[AgentType.Authority]=scores[self.authority_nodes]
        if dist_type=="all":
            return score_dict
        else:
            return score_dict[dist_type]

    def save_pickle(self,file_name):
        graph_var=self.graph
        self.graph=None
        pickle.dump(self,open(file_name,"wb"))
        nk.writeGraph(graph_var,file_name+".nk", nk.Format.NetworkitBinary)
        self.graph=graph_var

    @classmethod
    def load_pickle(cls,file_name):
        network = pickle.load(open(file_name,"rb"))
        graph=nk.readGraph(file_name+".nk",nk.Format.NetworkitBinary)
        network.graph=graph
        return network

    def get_nodes_by_status(self):
        node_status={}

        for node in self.Nodes.values():
            node_lists=node_status.get(node.news_state,[])
            node_lists.append(node)
            node_status[node.news_state]=node_lists

        return node_status


    def remove_edge(self,edge):
        self.graph.removeEdge(edge.agentfrom.id,edge.agentto.id)

    def add_edge(self,edge):
        self.graph.addEdge(edge.agentfrom.id,edge.agentto.id)

def calculate_engagement(node: Agent,twt):


    if twt.political:
        if node.political<0:
            tweet_int=-1*twt.political
            node_int=-1*node.political

        else:
            tweet_int=twt.political
            node_int=node.political
    else:
        tweet_int=max(twt.int_1,twt.int_2,twt.int_3)
        node_int=max(node.int_1,node.int_2,node.int_3)

    exp_factor=(-4-10*(1-abs(tweet_int)))*(node_int-1)
    return  (2*tweet_int)/(1+np.exp(exp_factor))


def check_if_faked(vul,trust,engagement):
    return random.uniform(0,1) < (trust+engagement)/2

def check_if_recovered(trust,engagement):
    return random.uniform(0,1) < (trust+engagement)/4


def set_tweets_to_edges(network:Network):
    for node in network.Nodes.values():
        for edge in node.followers:
            edge.tweets+=node.tweets
        node.tweets=[]

class Stage:
    def __init__(self, stage_no: int, n: Network):
        self.number=stage_no
        self.status_count={}
        self.edge_trust=[]
        self.tweets=[]
        self.infected_nodes=[]
        self.infected_node_polarity=[]
        self.political_statuses={}
        self.nodes_dict=[]
        self.disease_status={}
        self.bot_follow_percentage=[]
        self.node_types_per_status={}
        self.reproduction_no=[]
        self.political_polarity=[]
        self.political_engagement_polarity=[]
        self.political_engagement_polarity_med=[]
        self.degree_dist={}
        self.degree_dist_in={}
        self.contact=[]

        self.degree_dist=n.degree_distribution()
        self.degree_dist_in=n.degree_distribution(outdegree=False)


        #n.draw_political_network(f"network_{stage_no}",stage_no)

        for node in n.Nodes.values():
            self.contact.append(node.contact)
            self.nodes_dict.append(node.asdict())
            self.status_count[node.news_state]=self.status_count.get(node.news_state,0)+1
            self.disease_status[node.disease_state]=self.disease_status.get(node.disease_state,0)+1

            political_list=self.political_statuses.get(node.news_state,[])
            political_list.append(node.political)
            self.political_statuses[node.news_state]=political_list

            node_type_status=self.node_types_per_status.get(node.news_state,[])
            node_type_status.append(node.type)
            self.node_types_per_status[node.news_state]=node_type_status

            same_pol=0
            eng_list=[]
            for edge in node.following:
                self.edge_trust.append(edge.trust)
                self.tweets.append(len(edge.tweets))

                if edge.agentto.type in [AgentType.Regular,AgentType.Influencer]:
                    if edge.agentto.political * node.political >= 0 :
                        same_pol+=1
                    eng_list.append(calculate_engagement(node,edge.agentto))


            try:
                self.political_polarity.append(float(same_pol)/len(node.following))
            except ZeroDivisionError:
                pass
            else:
                self.political_engagement_polarity.append(np.mean(eng_list))
                self.political_engagement_polarity_med.append(np.median(eng_list))

            if node.news_state==FakeNewsState.Infected:
                self.infected_nodes.append(node.id)

            if node.infection_time!=-1:
                self.reproduction_no.append(len(node.transmitted_agents))


        for node in self.infected_nodes:
            infected_node=n.Nodes[node]

            infected_followers=0
            bot_followers=0

            for follower_edge in infected_node.following:
                if follower_edge.agentto.type==AgentType.Bot:
                    bot_followers+=1
                if follower_edge.agentto.id in self.infected_nodes:
                    infected_followers+=1
            try:
                self.infected_node_polarity.append(float(infected_followers)/ (len(infected_node.following)) )
                self.bot_follow_percentage.append(float(bot_followers)/len(infected_node.following))
            except ZeroDivisionError:
                pass

        ## If no node in state add zero
        for state in [FakeNewsState.Susceptible,FakeNewsState.Infected,FakeNewsState.Noninfectable,FakeNewsState.Recovered,FakeNewsState.Unfaked]:
            self.status_count[state]=self.status_count.get(state,0)
            self.political_statuses[state]=self.political_statuses.get(state,[])
            self.node_types_per_status[state]=self.node_types_per_status.get(state,[])

        for state in [DiseaseState.Infected,DiseaseState.Susceptible,DiseaseState.NonApplicable, DiseaseState.Recovered]:
            self.disease_status[state]=self.disease_status.get(state,0)

class SimulatorResults:
    dict_attr=["status_count","political_statuses","disease_status","node_types_per_status","degree_distribution","degree_dist_in"]
    list_attr=["tweets","edge_trust","infected_node_polarity","bot_follow_percentage","reproduction_no","political_polarity","political_engagement_polarity","political_engagement_polarity_med","contact"]


    def __init__(self,resuts):
        self.results=resuts

    def combine_results(self,attr,combine_func=np.mean,reduce_func=np.mean):

        if attr in self.dict_attr:
            overall_result={}
            for stage_list in self.results:
                stage_dict={}
                for stage in stage_list:
                    stage_attribute=stage.__getattribute__(attr)
                    for k,v in stage_attribute.items():
                        stage_dict_list=stage_dict.get(k,[])
                        stage_dict_list.append(v)
                        stage_dict[k]=stage_dict_list

                for k,v in stage_dict.items():
                    overall_result_list=overall_result.get(k,np.empty((0,len(v))))
                    overall_result_list=np.append(overall_result_list,np.array([v]),axis=0)
                    overall_result[k]=overall_result_list

            for k,v in overall_result.items():
                result=combine_func(v,axis=0)
                overall_result[k]=result

            return overall_result

        elif attr in self.list_attr:
            overall_results=np.empty((0,len(self.results[0])))
            for stage_list in self.results:
                comb_results=[]
                for stage in stage_list:
                    stage_attrib=stage.__getattribute__(attr)
                    comb_results.append(reduce_func(np.array(stage_attrib)))
                overall_results=np.append(overall_results,np.array([comb_results]),axis=0)

            return combine_func(overall_results,axis=0)


    def get_stage(self,attrib,stage_no=-1):
        if attrib in self.dict_attr:
            combined_results={}
            for stage_list in self.results:
                stage=stage_list[stage_no]
                attrib_dict=stage.__getattribute__(attrib)
                for k,v in attrib_dict.items():
                    combined_results[k]=combined_results.get(k,[])+v
            return combined_results
        elif attrib in self.list_attr:
            combined_results=[]
            for stage_list in self.results:
                stage=stage_list[stage_no]
                attrib=stage.__getattribute__(attrib)
                combined_results+=attrib

    def save(self,name):
        if name:
            pickle.dump(self,open(f"{name}.results","wb"))

class Simulator:

    @staticmethod
    def check_if_transmitted(agent1,agent2,agent_time):
        agent1.contact+=1
        agent2.contact+=1
        if agent1.disease_state==DiseaseState.Infected and agent2.disease_state==DiseaseState.Susceptible:
            if random.uniform(0,1)< agent1.health_precautions*agent2.health_precautions:
                agent2.disease_state=DiseaseState.Infected
                agent2.infection_time=agent_time
                agent2.health_precautions+=HEALTH_PRECAUTIONS_ALPHA
                agent1.transmitted_agents.append(agent2)

        elif agent2.disease_state==DiseaseState.Infected and agent1.disease_state==DiseaseState.Susceptible:
            if random.uniform(0,1)< agent1.health_precautions*agent2.health_precautions:
                agent1.disease_state=DiseaseState.Infected
                agent1.infection_time=agent_time
                agent1.health_precautions+=HEALTH_PRECAUTIONS_ALPHA
                agent2.transmitted_agents.append(agent1)

    @staticmethod
    def simulate_disease(agents,centralized_locations_dict :dict,stage_no:int,vaccine_stage_time=50):

        for agent in agents.values():
            agent.contact=0

        # Group Activity Stage
        for agent_groups in centralized_locations_dict.values():
            generator=nk.generators.WattsStrogatzGenerator(len(agent_groups),MEAN_CONTACT_GROUPED,0.1)
            contact_network=generator.generate()
            for u,v in contact_network.iterEdges():
                Simulator.check_if_transmitted(agent_groups[u],agent_groups[v],stage_no)


        # Random walk stage
        points_list=np.empty((0,2))
        for agent in agents.values():
            new_array=[min(1,max(0,agent.geographical[0]+normal(0,0.05,-1,1))),min(1,max(0,agent.geographical[1]+normal(0,0.05,-1,1)))]
            points_list=np.append(points_list,np.array([new_array]),axis=0)

        kdtree=KDTree(data=points_list)
        paired_trees=kdtree.query_pairs(r=CONTACT_MINIMUM_THRESHOLD)

        for u,v in paired_trees:
            Simulator.check_if_transmitted(agents[u],agents[v],stage_no)
        

        # Recovery stage
        for agent in agents.values():
            if agent.disease_state==DiseaseState.Infected and (agent.infection_time + agent.recovery_time)<stage_no:
                agent.disease_state=DiseaseState.Susceptible
                agent.transmitted_agents=[]
                agent.health_precautions+=HEALTH_PRECAUTIONS_ALPHA
                #agent.vaccine_rate+=VAXX_PRECAUTION_ALPHA

            if stage_no>50 and random.uniform(0,1)<agent.vaccine_rate:
                agent.disease_state=DiseaseState.Recovered

    @staticmethod
    def simulate_basic(n : Network, n_stages=100, no_disease=False,results_queue=None,save_name=None):
        network=copy.deepcopy(n)
        stages_list=[]
        for stage in range(n_stages):
            print(f"{stage}  ",end=" ")
            set_tweets_to_edges(network)
            stages_list.append(Stage(stage,network))

            if save_name and stage %5==0:
                pickle.dump(stages_list,open(save_name,"wb"))

            for node in network.Nodes.values():
                #CREATE NEW TWEET FIRST
                #Bots tweet at every stage.
                if node.type==AgentType.Bot:
                    new_tweet=Tweet(1,node,stage)
                    node.add_tweets(new_tweet)

                ## Check for access time for other accounts
                elif (stage-node.access_time)>=node.last_access:
                    node.last_access=stage
                    for edge in node.following:
                        #TODO: Add randomness in displaying tweets here
                        tweets=edge.tweets
                        edge.tweets=[]
                        for twt in tweets:
                            if random.uniform(0,1)< twt.calculate_prob(stage,node.access_time):
                                reshare_twt=twt
                                eng=calculate_engagement(node,twt)
                                if twt.misinformation and node.news_state==FakeNewsState.Susceptible:
                                    if check_if_faked(node.vul,edge.trust,eng):
                                        node.change_news_state( FakeNewsState.Infected)

                                        #node.faked_tweets.append(twt.id)
                                    else:
                                        reshare_twt=Tweet(0,node,stage,1,twt)
                                        node.change_news_state(FakeNewsState.Unfaked)

                                if twt.correction:

                                    if twt.tweeter.type==AgentType.Authority:
                                        recover_bool=check_if_recovered(node.est_trust,eng)
                                    else:
                                        recover_bool=check_if_recovered(edge.trust,eng)
                                    if (node.news_state==FakeNewsState.Infected):
                                        if recover_bool:
                                            #node.faked_tweets.remove(twt.id)
                                            # Check if node recovered from all the fake news
                                            #if not node.faked_tweets:
                                            node.change_news_state(FakeNewsState.Recovered)

                                ## Reshare tweet for non-political tweets requires a positive engagement threshold
                                if not reshare_twt.political:
                                    if random.uniform(0,1)<node.tweet_rate and eng> RETWEET_ENGAGEMENT_THRESHOLD:
                                        node.add_tweets(reshare_twt)
                                else:
                                    if random.uniform(0,1)<node.tweet_rate:
                                        node.add_tweets(reshare_twt)

                                edge.update_edge(eng)

                    ## New Tweet
                    if random.uniform(0,1) < node.tweet_rate:
                        new_tweet=Tweet(0,node,stage)
                        node.add_tweets(new_tweet)

            if not no_disease and stage>5:
                Simulator.simulate_disease(network.Nodes,network.centralized_locations,stage)
        if not results_queue==None:
            results_queue.append(stages_list)
        else:
            return stages_list


    @staticmethod
    def simulate_with_pol_change(n : Network, n_stages=100,no_disease=False,results_queue=None,save_name=None,drop_edge=True):
        network=copy.deepcopy(n)
        stages_list=[]
        for stage in range(n_stages):
            print(f"{stage}  ",end=" ")
            set_tweets_to_edges(network)
            stages_list.append(Stage(stage,network))
            if save_name and stage %5==0:
                pickle.dump(stages_list,open(save_name,"wb"))

            for node in network.Nodes.values():
                #CREATE NEW TWEET FIRST
                #Bots tweet at every stage.
                if node.type==AgentType.Bot:
                    new_tweet=Tweet(1,node,stage)
                    node.add_tweets(new_tweet)

                ## Check for access time for other accounts
                elif (stage-node.access_time)>=node.last_access:
                    node.last_access=stage
                    for edge in node.following:
                        #TODO: Add randomness in displaying tweets here
                        tweets=edge.tweets
                        edge.tweets=[]
                        for twt in tweets:
                            if random.uniform(0,1)< twt.calculate_prob(stage,node.access_time):
                                reshare_twt=twt
                                eng=calculate_engagement(node,twt)

                                if drop_edge and eng>FOLLOW_ENGAGEMENT_THRESHOLD:
                                    new_edge=node.follow(twt.tweeter)
                                    if new_edge:
                                        network.add_edge(new_edge)
                                        if new_edge.agentto.id!=twt.tweeter.id or new_edge.agentfrom.id != node.id:
                                            raise Exception("This isn't supposed to happen yo")

                                if twt.political:
                                    node.update_political(eng)

                                if twt.misinformation and node.news_state==FakeNewsState.Susceptible:
                                    if check_if_faked(node.vul,edge.trust,eng):
                                        node.change_news_state( FakeNewsState.Infected)
                                        #node.faked_tweets.append(twt.id)

                                    else:
                                        reshare_twt=Tweet(0,node,stage,1,twt)
                                        node.change_news_state(FakeNewsState.Unfaked)


                                if twt.correction:

                                    if twt.tweeter.type==AgentType.Authority:
                                        recover_bool=check_if_recovered(node.est_trust,eng)
                                    else:
                                        recover_bool=check_if_recovered(edge.trust,eng)


                                    if (node.news_state==FakeNewsState.Infected):
                                        if recover_bool:
                                            #node.faked_tweets.remove(twt.id)
                                            # Check if node recovered from all the fake news
                                            #if not node.faked_tweets:
                                            node.change_news_state(FakeNewsState.Recovered)

                                ## Reshare tweet for non-political tweets requires a positive engagement threshold
                                if not reshare_twt.political:
                                    if random.uniform(0,1)<node.tweet_rate and eng> RETWEET_ENGAGEMENT_THRESHOLD:
                                        node.add_tweets(reshare_twt)
                                else:
                                    if random.uniform(0,1)<node.tweet_rate:
                                        node.add_tweets(reshare_twt)

                                edge.update_edge(eng)

                        if drop_edge and edge.trust<0.45:
                            node.unfollow(edge)
                            network.remove_edge(edge)

                    ## New Tweet
                    if random.uniform(0,1) < node.tweet_rate:
                        new_tweet=Tweet(0,node,stage)
                        node.add_tweets(new_tweet)

            if not no_disease and stage>5:
                Simulator.simulate_disease(network.Nodes,network.centralized_locations,stage)

        if not results_queue == None :
            results_queue.append(stages_list)
        else:
            return stages_list


    def __init__(self,network: Network, n_repeated_sim=1):
        if not network.finalized:
            network.add_edges(0.1,0.1,0.02,0.1)
        self.network=network
        self.repetitions=n_repeated_sim
        self.results=[]


    def run(self,n_stages=100,return_results=False,save_name=None,save_stages=False,no_misinformation=False, no_disease=False,political_change=True):
        saven=None
        if not save_name:
            save_name="simulator"
        if save_stages:
            saven=f"{save_name}_{i}.stage"
        n=copy.deepcopy(self.network)

        if no_misinformation:
            stages_list=[]
            for stage_no in range(n_stages):
                Simulator.simulate_disease(n.Nodes,n.centralized_locations,stage_no)
                stages_list.append(Stage(stage_no,n))
            self.results.append(stages_list)

        else:
            """
            processes=[]
            m=mp.Manager()
            q=m.list()
            if political_change:
                for i in range(self.repetitions):
                    p=mp.Process(target=self.simulate_with_pol_change,args=(n,n_stages,no_disease,q))
                    processes.append(p)
                    p.start()
            else:
                for i in range(self.repetitions):
                    p=mp.Process(target=self.simulate_basic,args=(n,n_stages,no_disease,q) )
                    processes.append(p)
                    p.start()

            for p in processes:
                p.join()

            self.results=list(q)
            """
            for i in range(self.repetitions):
                if political_change:
                    self.results.append(self.simulate_basic(n,n_stages,no_disease))
                else:
                    self.results.append(self.simulate_with_pol_change(n,n_stages,no_disease))

        if return_results:
            return self.results

        results=SimulatorResults(self.results)
        results.save(save_name)


if __name__=="__main__":
    """
    ## Initial configuration
    n=Network(1000,20,20,0,0,10,0)
    n.add_edges(0.05,0.05,0.01,0)
    sim=Simulator(n,10)
    sim.run(100,save_name="nopol_init",no_disease=True,political_change=False)
    """

    for n_bots in [10,20,40,80]:
        n=Network(1000,20,n_bots,0,0,10,0)
        n.add_edges(0.05,0.05,0.01,0)
        sim=Simulator(n,10)
        sim.run(50,save_name=f"nopol_changing_bots_{n_bots}",no_disease=True,political_change=False)